"""Central runtime configuration for Arca.

Every deployment-specific name (Unity Catalog, Vector Search endpoint/index)
and tunable (similarity threshold, deadlines, cache sizes) is env-driven with
the ``ARCA_`` prefix, so nobody has to edit source to point Arca at their own
workspace. Databricks credentials (DATABRICKS_HOST / DATABRICKS_TOKEN /
DATABRICKS_HTTP_PATH) are read by the Databricks SDKs directly and are
deliberately not duplicated here.

``ARCA_CACHE_ENABLED`` is intentionally NOT part of this snapshot: it must be
re-read from the environment on every request (see arca.cache._cache_enabled)
so the cache can be toggled without restarting the proxy.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARCA_", extra="ignore")

    # --- Databricks namespace -------------------------------------------
    catalog: str = "main"
    db_schema: str = Field(
        default="arca",
        validation_alias=AliasChoices("ARCA_SCHEMA", "db_schema"),
    )
    vs_endpoint: str = "arca-vs-endpoint"

    # --- Cache behaviour --------------------------------------------------
    # 0.90 is safe ONLY because the L2 polarity guard (arca.semantic_guard)
    # rejects direction/negation flips that leak in [0.90,0.95). Measured: cosine
    # 0.90 + guard = 92% precision / 61% recall held-out, vs 75% precision at 0.95
    # cosine-only. Do not raise this without the guard. See benchmarks/EVAL_METRICS.md.
    similarity_threshold: float = 0.90
    l1_max_entries: int = 1024
    l2_deadline_s: float = 0.5

    # Local semantic fallback: brute-force cosine over the SQLite store when
    # Databricks Vector Search is not configured. Lets Arca run standalone.
    local_l2: bool = True
    local_l2_max_rows: int = 5000

    # --- Proxy ------------------------------------------------------------
    port: int = 8082

    @property
    def cache_table(self) -> str:
        return f"{self.catalog}.{self.db_schema}.cache_store"

    @property
    def usage_table(self) -> str:
        return f"{self.catalog}.{self.db_schema}.usage_log"

    @property
    def vs_index(self) -> str:
        return f"{self.catalog}.{self.db_schema}.prompt_index"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings snapshot. Call ``get_settings.cache_clear()``
    in tests after monkeypatching env vars."""
    return Settings()
