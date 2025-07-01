"""
The InferenceEngine is responsible for managing vLLM servers and orchestrating
requests to the LLM agents.
"""
import subprocess
import time
import requests
from typing import List, Tuple
from .allocator import RoundRobinAllocator

import pynvml
pynvml.nvmlInit()

import os

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

        self.gpu_info = []
        
        print("Initializing Inference Engine...")
        available_lanes = self._discover_and_launch_servers()
        
        if not available_lanes:
            raise RuntimeError("Failed to launch any vLLM servers. Check GPU availability and drivers.")

        self.allocator = RoundRobinAllocator(available_lanes)
        print(f"Inference Engine ready. Managing {len(available_lanes)} agent lanes.")


    def get_vllm_launch_command(self, gpu_id: int, port: int) -> list:
        """
        Returns the vLLM server launch command for a given GPU and port.
        """
        return [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--host", "localhost",
                "--port", str(port),
                "--tensor-parallel-size", "1", # Each server is self-contained
                # Removed --device call because we set it in the env["CUDA...""]
                "--model", "/PATH/TO/MODEL"
            ]

    def isMPS(self, gpu_id: int) -> bool:
        """
        Takes an index of a GPU and detects whether or not it supports MPS.

        Returns:
            A boolean whether MPS is enabled or not.
        """
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "-i", str(gpu_id), "-q"],
                text=True
            )
            return "MPS" in output
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to check MPS for GPU {gpu_id}.")
            return False
        
    def start_mps_daemon(self, pipe_dir, log_dir):
        os.makedirs(pipe_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
        env["CUDA_MPS_LOG_DIRECTORY"] = log_dir
        try:
            subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)
            print("MPS daemon started.")
        except subprocess.CalledProcessError:
            print("MPS daemon may already be running.")

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
            gpu_count = pynvml.nvmlDeviceGetCount()

            # Get indices
            for gpu_id in range(gpu_count):
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                gpu_name = pynvml.nvmlDeviceGetName(gpu_handle).decode()
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)

                total_mb = meminfo.total // 1024**2 # Convert from bytes to mb
                free_mb = meminfo.free // 1024**2

                isMPS = self.isMPS(gpu_id)

                self.gpu_info.append((gpu_id, gpu_name, total_mb, free_mb, isMPS))

            print(f"Discovered {gpu_count} GPUs.")

        except (FileNotFoundError, subprocess.CalledProcessError):
            print("Warning: `nvidia-smi` not found. Assuming 0 GPUs.")
            gpu_count = 0
            return []

        lanes = []
        port_offset = 0
        port_base = 8000
        thread_split = [int(100 / self.models_per_gpu)] * self.models_per_gpu

        pipe_dir = f"/tmp/nvidia-mps-{os.environ['USER']}"
        log_dir = f"/tmp/nvidia-mps-log-{os.environ['USER']}"

        # Check if any GPU needs MPS — and if so, start the daemon once
        any_mps = any(info[4] for info in self.gpu_info)
        if any_mps:
            if not os.path.exists(os.path.join(pipe_dir, "nvidia-mps")):
                print("Starting user-level MPS daemon...")
                self.start_mps_daemon(pipe_dir, log_dir)
            else:
                print("MPS daemon appears to be already running.")

        for gpu_id, _, _, _, is_mps in self.gpu_info:
            if is_mps:
                print(f"GPU {gpu_id}: MPS Supported")

                for pct in thread_split:
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    env["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
                    env["CUDA_MPS_LOG_DIRECTORY"] = log_dir
                    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(pct)

                    current_port = port_base + port_offset
                    command = self.get_vllm_launch_command(gpu_id, current_port)

                    print(f"Launching vLLM server on GPU {gpu_id} at port {current_port}...")
                    server_process = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    self.servers.append(server_process)
                    lanes.append((gpu_id, f"http://localhost:{current_port}"))
                    port_offset += 1

            else:
                print(f"GPU {gpu_id}: MPS NOT supported — launching normal agent")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                current_port = port_base + port_offset
                command = self.get_vllm_launch_command(gpu_id, current_port)

                print(f"Launching vLLM server on GPU {gpu_id} at port {current_port}...")
                server_process = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                self.servers.append(server_process)
                lanes.append((gpu_id, f"http://localhost:{current_port}"))
                port_offset += 1

                
        # Give servers a moment to initialize
        # A more robust implementation would poll the server endpoints until they are ready.
        time.sleep(10)
        pynvml.nvmlShutdown()
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
