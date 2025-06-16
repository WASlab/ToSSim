"""
The InferenceEngine is responsible for managing vLLM servers and orchestrating
requests to the LLM agents.
"""
import subprocess
import time
from typing import List, Tuple
from .allocator import RoundRobinAllocator

class InferenceEngine:
    """
    Manages vLLM server instances and handles the lifecycle of agent requests.
    """
    def __init__(self, models_per_gpu: int = 2):
        """
        Initializes the InferenceEngine.

        This process involves:
        1. Discovering available NVIDIA GPUs.
        2. Launching a specified number of vLLM server instances for each GPU.
        3. Initializing the resource allocator with the server details.
        """
        self.models_per_gpu = models_per_gpu
        self.servers: List[subprocess.Popen] = []
        
        print("Initializing Inference Engine...")
        available_lanes = self._discover_and_launch_servers()
        
        if not available_lanes:
            raise RuntimeError("Failed to launch any vLLM servers. Check GPU availability and drivers.")

        self.allocator = RoundRobinAllocator(available_lanes)
        print(f"Inference Engine ready. Managing {len(available_lanes)} agent lanes.")

    def _discover_and_launch_servers(self) -> List[Tuple[int, str]]:
        """
        Discovers GPUs and launches vLLM servers.

        Returns:
            A list of (gpu_id, server_url) tuples for the allocator.
        """
        # In a real implementation, we would use pynvml or parse nvidia-smi.
        # For this skeleton, we'll assume 2 GPUs are available.
        try:
            # A simple way to check for nvidia-smi's existence and get GPU count
            gpu_info = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8")
            gpu_count = len(gpu_info.strip().split("\n"))
            print(f"Discovered {gpu_count} GPUs.")
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("Warning: `nvidia-smi` not found. Assuming 0 GPUs.")
            gpu_count = 0
            return []

        lanes = []
        current_port = 8000
        for gpu_id in range(gpu_count):
            for _ in range(self.models_per_gpu):
                # This command is an example. It would need to be configured
                # with the correct model path for the agents.
                command = [
                    "python", "-m", "vllm.entrypoints.openai.api_server",
                    "--host", "localhost",
                    "--port", str(current_port),
                    "--tensor-parallel-size", "1", # Each server is self-contained
                    f"--device", str(gpu_id)
                    # The --model argument would be added here later when registering
                    # specific agents, or pre-loaded if all agents use the same base.
                ]
                
                print(f"Launching vLLM server on GPU {gpu_id} at port {current_port}...")
                # We launch the server as a background process
                server_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                self.servers.append(server_process)
                
                lanes.append((gpu_id, f"http://localhost:{current_port}"))
                current_port += 1
        
        # Give servers a moment to initialize
        # A more robust implementation would poll the server endpoints until they are ready.
        time.sleep(10) 
        
        return lanes

    def register_agent(self, agent_id: str, model_checkpoint: str):
        """
        Assigns an agent to a GPU lane and potentially loads its model.
        (Implementation to follow based on docs)
        """
        lane = self.allocator.get_lane(agent_id)
        print(f"Agent '{agent_id}' assigned to GPU {lane[0]} on {lane[1]}.")
        # Here you would send a request to the vLLM server at lane[1] to
        # ensure the specified model_checkpoint is loaded.

    def shutdown(self):
        """
        Terminates all running vLLM server processes.
        """
        print("Shutting down all vLLM servers...")
        for server in self.servers:
            server.terminate()
        for server in self.servers:
            server.wait()
        print("Shutdown complete.")
