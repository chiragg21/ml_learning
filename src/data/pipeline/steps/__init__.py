from __future__ import annotations

import importlib
import pkgutil

# Auto-discovery: importing this package imports every module placed
# alongside it, so a new step file only needs to define a function and
# decorate it with @StepRegistry.register("name") - no manual wiring here.
for _module_info in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_module_info.name}")
