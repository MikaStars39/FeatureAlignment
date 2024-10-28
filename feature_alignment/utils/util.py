from importlib import import_module
from omegaconf import DictConfig

def instantiate(config: DictConfig, instantiate_module=True):
    """Get arguments from config."""
    module = import_module(config.module_name)
    class_ = getattr(module, config.class_name)
    if instantiate_module:
        init_args = {k: v for k, v in config.items() if k not in ["module_name", "class_name"]}
        return class_(**init_args)
    else:
        return class_

