import time
import torch
import logging
import datetime
from collections import OrderedDict
from contextlib import contextmanager
from detectron2.utils.comm import is_main_process
from .calibration_layer import PrototypicalCalibrationBlock

class DatasetEvaluator:
    def reset(self):
        pass

    def process(self, input, output):
        pass

    def evaluate(self):
        pass

class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process():
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results

def inference_on_dataset(model, data_loader, evaluator, cfg=None):
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    pcb = None
    if cfg.TEST.PCB_ENABLE:
        logger.info("Start initializing PCB module, please wait a seconds...")
        pcb = PrototypicalCalibrationBlock(cfg)
    logger.info("Start inference on {} images".format(len(data_loader)))
    total = len(data_loader)
    evaluator.reset()
    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0
            start_compute_time = time.time()
            outputs = model(inputs)
            if cfg.TEST.PCB_ENABLE:
                outputs = pcb.execute_calibration(inputs, outputs)
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            evaluator.process(inputs, outputs)
            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    results = evaluator.evaluate()
    if results is None:
        results = {}
    return results

@contextmanager
def inference_context(model):
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)