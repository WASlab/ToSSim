"""inference.allocator – simple agent→lane assignment

A *lane* is a tuple ``(gpu_id, url, port)`` corresponding to one vLLM server
slot (process) pinned to a fixed CUDA device and TCP port.  We keep exactly
one live vLLM process per lane and map **one agent id** to **one lane** for the
whole lifetime of a match.

Design requirements
-------------------
* Acquire an unused lane when a NEW agent id appears.
* Re-use the same lane if the agent id re-appears in the next match.
* Release the lane only after the agent's match ends so it can be recycled.
* No model-level sharing or eviction logic – every agent owns its own vLLM
  process even if multiple agents load the same checkpoint.
"""

from collections import deque
from threading import Lock
from typing import Deque, Dict, List, Tuple

Lane = Tuple[int, str]  # (gpu_id, "http://host:port")


class AgentAllocator:
    """Minimal lane allocator with stable agent→lane binding."""

    def __init__(self, available_lanes: List[Lane]):
        if not available_lanes:
            raise ValueError("Allocator needs at least one lane.")

        self._free: Deque[Lane] = deque(available_lanes)
        self._agent_to_lane: Dict[str, Lane] = {}
        self._lock: Lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self, agent_id: str) -> Lane:
        """Return a lane for *agent_id*.

        If the agent already holds a lane we return it, otherwise we pop the
        next free lane (FIFO).  Raises *RuntimeError* if no lane is available.
        """
        with self._lock:
            if agent_id in self._agent_to_lane:
                return self._agent_to_lane[agent_id]

            try:
                lane = self._free.popleft()
            except IndexError as exc:
                raise RuntimeError("No free GPU lanes left – cannot assign new agent") from exc

            self._agent_to_lane[agent_id] = lane
            return lane

    def release(self, agent_id: str) -> None:
        """Mark *agent_id*'s lane as free again (called at end of match)."""
        with self._lock:
            lane = self._agent_to_lane.pop(agent_id, None)
            if lane is not None:
                self._free.append(lane) 