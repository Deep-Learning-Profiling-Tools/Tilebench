"""Dynamic module loader for LLM-generated kernel files.

Allows the TileBench engine to load either the canonical hand-written
implementation or an LLM-generated override without altering the original
``benchmarks/operators/`` directory.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_module_from_file(module_name: str, path: str | Path) -> ModuleType:
    """Load a Python module from an arbitrary file path.

    The module is registered in ``sys.modules`` under *module_name* so that
    subsequent ``import`` statements within the loaded code resolve correctly.

    Parameters
    ----------
    module_name:
        Logical name to register in ``sys.modules``.
    path:
        Absolute or relative path to the ``.py`` file.

    Returns
    -------
    ModuleType
        The loaded module object.

    Raises
    ------
    ImportError
        If the spec cannot be created or the loader is ``None``.
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Module file not found: {path}")

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for '{module_name}' at {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def load_backend_module(
    operator_name: str,
    backend: str,
    override_path: str | Path | None = None,
) -> ModuleType:
    """Load the implementation module for a given operator and backend.

    When *override_path* is ``None``, falls back to the canonical import path
    ``benchmarks.operators.<operator_name>.impl_<backend>``.

    Parameters
    ----------
    operator_name:
        Name of the operator (e.g. ``"softmax"``).
    backend:
        Backend identifier: ``"triton"`` or ``"cutile"``.
    override_path:
        Path to an LLM-generated ``impl_<backend>.py`` file.  When supplied,
        the file is loaded directly instead of the canonical module.

    Returns
    -------
    ModuleType
        The loaded implementation module.
    """
    if override_path is None:
        return importlib.import_module(
            f"benchmarks.operators.{operator_name}.impl_{backend}"
        )

    unique_name = f"llm_generated_{operator_name}_{backend}"
    return load_module_from_file(unique_name, override_path)
