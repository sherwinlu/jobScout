from __future__ import annotations

from jobscout.config import ScoutConfig
from jobscout.llm import LocalLLMClient
from jobscout.storage import Storage


class BaseAgent:
    """Shared dependencies for JobScout agents."""

    name: str = "agent"

    def __init__(
        self,
        config: ScoutConfig,
        storage: Storage,
        llm: LocalLLMClient | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.llm = llm or LocalLLMClient.from_config(config.llm)
