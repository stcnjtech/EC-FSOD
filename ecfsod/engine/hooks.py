import os
import json
import time
import torch
import itertools
import detectron2.utils.comm as comm
from detectron2.config import global_cfg
from fvcore.common.file_io import PathManager
from detectron2.engine.train_loop import HookBase
from detectron2.evaluation.testing import flatten_results_dict

__all__ = ["EvalHookECFSOD"]

class EvalHookECFSOD(HookBase):
    def __init__(self, eval_period, eval_function, cfg):
        self._period = eval_period
        self._func = eval_function
        self.cfg = cfg

    def _do_eval(self):
        results = self._func()
        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)
            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)
        if comm.is_main_process() and results:
            is_final = self.trainer.iter + 1 >= self.trainer.max_iter
            os.makedirs(
                os.path.join(self.cfg.OUTPUT_DIR, 'inference'), exist_ok=True)
            output_file = 'res_final.json' if is_final else \
                'iter_{:07d}.json'.format(self.trainer.iter)
            with PathManager.open(os.path.join(self.cfg.OUTPUT_DIR, 'inference',
                                               output_file), 'w') as fp:
                json.dump(results, fp)
        comm.synchronize()

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            self._do_eval()

    def after_train(self):
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._do_eval()
        del self._func