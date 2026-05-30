"""arca.databricks package."""
import importlib as _importlib

# 00_bootstrap.py is not a valid Python identifier; load via importlib and
# expose as _bootstrap_impl so arca.databricks.bootstrap can reuse its helpers.
_bootstrap_impl = _importlib.import_module("arca.databricks.00_bootstrap")
