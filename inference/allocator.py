"""
GPU Resource Allocator

This module provides a simple mechanism for allocating agents to specific GPU
resources (vLLM server instances) in a round-robin fashion.
"""

from typing import List, Tuple, Dict

class RoundRobinAllocator:
    """
    Assigns agents to GPU server lanes in a round-robin cycle.

    This ensures that agents are distributed as evenly as possible across the
    available hardware resources.
    """
    def __init__(self, available_lanes: List[Tuple[int, str]]):
        """
        Initializes the allocator with a list of available GPU lanes.

        Args:
            available_lanes: A list of tuples, where each tuple represents a
                             lane and contains (gpu_id, server_url).
                             Example: [(0, "http://localhost:8000"), (0, "http://localhost:8001")]
        """
        if not available_lanes:
            raise ValueError("Cannot initialize an allocator with no available lanes.")
        
        self.lanes: List[Tuple[int, str]] = available_lanes
        self.next_lane_index: int = 0
        self.agent_assignments: Dict[str, Tuple[int, str]] = {}

    def get_lane(self, agent_id: str) -> Tuple[int, str]:
        """
        Gets the next available GPU lane for a given agent.

        If the agent has already been assigned a lane, it returns the existing
        assignment. Otherwise, it assigns the next lane in the cycle.

        Args:
            agent_id: The unique identifier for the agent.

        Returns:
            A tuple containing the (gpu_id, server_url) for the assigned lane.
        """
        if agent_id in self.agent_assignments:
            return self.agent_assignments[agent_id]

        # Get the next lane in the cycle
        assigned_lane = self.lanes[self.next_lane_index]
        self.agent_assignments[agent_id] = assigned_lane

        # Move to the next lane for the subsequent call, wrapping around if necessary
        self.next_lane_index = (self.next_lane_index + 1) % len(self.lanes)

        return assigned_lane 