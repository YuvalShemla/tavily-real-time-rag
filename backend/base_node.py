from __future__ import annotations
import abc, logging
from typing import Any, Dict

_log = logging.getLogger("backend.base_node")

class BaseNode(abc.ABC):
    """
    Every LangGraph node subclasses this and just implements run().
    __call__ adds entry/exit logging.
    """

    def __init__(self, name: str):
        self.name = name

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        _log.info("➡️  %s enter | keys=%s", self.name, list(state.keys()))
        new_state = await self.run(state)
        _log.info("⬅️  %s leave | keys=%s", self.name, list(new_state.keys()))
        return new_state

    @abc.abstractmethod
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ...
