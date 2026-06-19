"""arca.databricks package.

``bootstrap_impl`` is imported lazily (via ``__getattr__``) and exposed under the
legacy ``_bootstrap_impl`` alias. The laziness matters: importing this package
must NOT import bootstrap_impl, which (transitively) touches heavy SDKs — every
CLI invocation imports this package.
"""
import importlib as _importlib


def __getattr__(name):
    if name in ("bootstrap_impl", "_bootstrap_impl"):
        return _importlib.import_module("arca.databricks.bootstrap_impl")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
