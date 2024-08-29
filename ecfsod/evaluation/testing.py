import sys
import pprint
import logging
import numpy as np
from collections import Mapping, OrderedDict

def print_csv_format(results):
    assert isinstance(results, OrderedDict), results
    logger = logging.getLogger(__name__)
    for task, res in results.items():
        important_res = [(k, v) for k, v in res.items() if "-" not in k]
        logger.info("copypaste: Task: {}".format(task))
        logger.info("copypaste: " + ",".join([k[0] for k in important_res]))
        logger.info("copypaste: " + ",".join(["{0:.4f}".format(k[1]) for k in important_res]))

def verify_results(cfg, results):
    expected_results = cfg.TEST.EXPECTED_RESULTS
    if not len(expected_results):
        return True
    ok = True
    for task, metric, expected, tolerance in expected_results:
        actual = results[task][metric]
        if not np.isfinite(actual):
            ok = False
        diff = abs(actual - expected)
        if diff > tolerance:
            ok = False
    logger = logging.getLogger(__name__)
    if not ok:
        logger.error("Result verification failed!")
        logger.error("Expected Results: " + str(expected_results))
        logger.error("Actual Results: " + pprint.pformat(results))
        sys.exit(1)
    else:
        logger.info("Results verification passed.")
    return ok

def flatten_results_dict(results):
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_results_dict(v)
            for kk, vv in v.items():
                r[k + "/" + kk] = vv
        else:
            r[k] = v
    return r