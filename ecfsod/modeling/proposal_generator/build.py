from detectron2.utils.registry import Registry

PROPOSAL_GENERATOR_REGISTRY = Registry("PROPOSAL_GENERATOR")

from . import rpn  # noqa F401 isort:skip

def build_proposal_generator(cfg, input_shape):
    name = cfg.MODEL.PROPOSAL_GENERATOR.NAME
    return PROPOSAL_GENERATOR_REGISTRY.get(name)(cfg, input_shape)