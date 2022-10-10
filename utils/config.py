import importlib
import sys
from typing import Any, List, Type

import yaml


KEY_MODULES = "MODULES"
KEY_CLASS = "CLASS"
KEY_REF = "REF"


def find_class(modules: List[str], class_name: str) -> Type:
    class_modules = class_name.split('.')
    if len(class_modules) > 1:
        class_name = class_modules[-1]
        class_modules = ".".join(class_modules[:-1])
    else:
        class_modules = ""

    for module_name in modules:
        if len(module_name) > 0 and len(class_modules) > 0:
            full_module_name = module_name + "." + class_modules
        else:
            full_module_name = module_name + class_modules
        module = sys.modules.get(full_module_name)
        if module is None:
            module = importlib.import_module(full_module_name)
        if hasattr(module, class_name):
            return getattr(module, class_name)

    raise RuntimeError(f"Cannot find class {class_name}")


def resolve_classes(config: Any, modules: List[str], parent_path: List[Any]) -> Any:
    if isinstance(config, dict):
        if KEY_REF in config:
            assert 1 == len(config)
            ref_name = config[KEY_REF]
            for parent in reversed(parent_path[:-1]):
                if ref_name in parent:
                    return parent[ref_name]
            raise RuntimeError(f"Could not resolve reference {ref_name}!")

        class_name = config[KEY_CLASS] if KEY_CLASS in config.keys() else None
        if class_name is not None:
            del config[KEY_CLASS]

        # Note: Can't use list comprehension: objects can refer to already resolved objects.
        for k, v in config.items():
            config[k] = resolve_classes(v, modules, parent_path + [config])

        if class_name is not None:
            class_type = find_class(modules, class_name)
            return class_type(**config)

        return config

    if isinstance(config, list):
        return [resolve_classes(v, modules, parent_path + [config]) for v in config]

    return config


def extract_modules(config: dict) -> List[str]:
    if KEY_MODULES not in config.keys():
        return []

    modules = config[KEY_MODULES]
    del config[KEY_MODULES]

    return modules


def load_config(yaml_file: str) -> dict:
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    modules = extract_modules(config)
    config = resolve_classes(config, modules, [])

    return config
