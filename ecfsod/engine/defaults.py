import os
import torch
import logging
import argparse
from ecfsod.data import *
from detectron2.utils import comm
from collections import OrderedDict
from ecfsod.modeling import build_model
from detectron2.data import transforms as T
from fvcore.common.file_io import PathManager
from detectron2.utils.env import seed_all_rng
from ecfsod.engine.hooks import EvalHookECFSOD
from fvcore.nn.precise_bn import get_bn_modules
from detectron2.utils.logger import setup_logger
from detectron2.engine import hooks, SimpleTrainer
from ecfsod.checkpoint import DetectionCheckpointer
from torch.nn.parallel import DistributedDataParallel
from detectron2.utils.collect_env import collect_env_info
from ecfsod.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import TensorboardXWriter, CommonMetricPrinter, JSONWriter
from ecfsod.evaluation import DatasetEvaluator, inference_on_dataset, print_csv_format, verify_results
from ecfsod.dataloader import MetadataCatalog, build_detection_test_loader, build_detection_train_loader

__all__ = [
    "default_argument_parser",
    "default_setup",
    "DefaultPredictor",
    "DefaultTrainer",
]

def default_argument_parser():
    parser = argparse.ArgumentParser(description="EC-FSOD Training")
    parser.add_argument("--config-file", default="", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--resume", action="store_true",
                        help="whether to attempt to resume")
    parser.add_argument("--eval-only", action="store_true",
                        help="evaluate last checkpoint")
    parser.add_argument("--eval-all", action="store_true",
                        help="evaluate all saved checkpoints")
    parser.add_argument("--eval-during-train", action="store_true",
                        help="evaluate during training")
    parser.add_argument("--eval-iter", type=int, default=-1,
                        help="checkpoint iteration for evaluation")
    parser.add_argument("--start-iter", type=int, default=-1,
                        help="starting iteration for evaluation")
    parser.add_argument("--end-iter", type=int, default=-1,
                        help="ending iteration for evaluation")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0,
                        help="the rank of this machine")
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    return parser

def default_setup(cfg, args):
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)
    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    setup_logger(output_dir, distributed_rank=rank, name="ecfsod")
    logger = setup_logger(output_dir, distributed_rank=rank)
    logger.info(
        "Rank of current process: {}. World size: {}".format(
            rank, comm.get_world_size()
        )
    )
    if not cfg.MUTE_HEADER:
        logger.info("Environment info:\n" + collect_env_info())
    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file"):
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                PathManager.open(args.config_file, "r").read(),
            )
        )
    if not cfg.MUTE_HEADER:
        logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK

class DefaultPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST,
        )
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    @torch.no_grad()
    def __call__(self, original_image):
        if self.input_format == "RGB":
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = self.transform_gen.get_transform(original_image).apply_image(
            original_image
        )
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        predictions = self.model([inputs])[0]
        return predictions

class DefaultTrainer(SimpleTrainer):
    def __init__(self, cfg):
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        super().__init__(model, data_loader, optimizer)
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        self.start_iter = (
            self.checkpointer.resume_or_load(
                self.cfg.MODEL.WEIGHTS, resume=resume
            ).get("iteration", -1)
            + 1
        )

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = (0)
        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )
        
        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(EvalHookECFSOD(
            cfg.TEST.EVAL_PERIOD, test_and_save_results, self.cfg))
        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers()))
        return ret

    def build_writers(self):
        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    def train(self):
        super().train(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        if not cfg.MUTE_HEADER:
            logger = logging.getLogger(__name__)
            logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        raise NotImplementedError(
            "Please either implement `build_evaluator()` in subclasses, or pass "
            "your evaluator as arguments to `DefaultTrainer.test()`."
        )

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(
                evaluators
            ), "{} != {}".format(len(cfg.DATASETS.TEST), len(evaluators))
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator, cfg)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info(
                    "Evaluation results for {} in csv format:".format(
                        dataset_name
                    )
                )
                print_csv_format(results_i)
        if len(results) == 1:
            results = list(results.values())[0]
        return results