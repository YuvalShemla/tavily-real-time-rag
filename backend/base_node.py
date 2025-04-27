"""
    Base node — abstract wrapper every LangGraph node extends.
    Handles front end streaming, and entry/exit logging; subclasses need to implement run().
"""

# base_node.py
from __future__ import annotations
import abc, logging
from typing import Any, Dict
_log = logging.getLogger("backend.base_node")

# --------- Base Node ---------
class BaseNode(abc.ABC):
    """Abstract wrapper that adds uniform logging around `run()`."""

    def __init__(self, name: str):
        self.name = name

    # log graph transitions
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Log entry and exit from this node, and call `run()`."""
        
        _log.info(" entering %s", self.name)

        new_state = await self.run(state)

        _log.info(" leaving  %s", self.name)
 
        return new_state


    @abc.abstractmethod
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node logic; return an update to the graph state."""
