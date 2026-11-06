"""Load the external package without adding anything to sys.path."""
from pathlib import Path
import importlib.util
import sys


def install(app_dir):
    name = "stella_sora_value_plugin"
    package = sys.modules.get(name)
    if package is None:
        root = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
    return package.install(app_dir)
