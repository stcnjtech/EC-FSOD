from detectron2.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")

def build_model(cfg):
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    return META_ARCH_REGISTRY.get(meta_arch)(cfg)