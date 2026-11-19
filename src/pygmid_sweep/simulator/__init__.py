"""pygmid sweep simulators"""

from __future__ import annotations

import importlib
import os
from glob import glob
from pathlib import Path

# Files to import first, relative to this package
import_first: list[str] = ["sim.py"]

# Find all .py files in this directory except __init__.py
all_py = [
    f
    for f in glob(
        "*.py",
        root_dir=Path(__file__).parent.absolute(),
    )
    if not f.startswith("_")
]

# Order imports: import_first first, then the rest
ordered = []
for f in import_first:
    if f in all_py:
        ordered.append(f)
for f in all_py:
    if f not in import_first:
        ordered.append(f)

__all__: list[str] = []

for fname in ordered:
    module_name = os.path.splitext(fname)[0]
    module = importlib.import_module(f".{module_name}", package=__name__)
    # expose public names defined by the module
    if hasattr(module, "__all__"):
        globals().update({k: getattr(module, k) for k in module.__all__})
        __all__.extend(module.__all__)
    else:
        # if no __all__, expose all public attributes
        for attr in dir(module):
            if not attr.startswith("_"):
                globals()[attr] = getattr(module, attr)
                __all__.append(attr)
