"""
The InferenceEngine is responsible for managing vLLM servers and orchestrating
requests to the LLM agents.
"""
import subprocess
import time
import requests
from typing import List, Tuple
from .allocator import AgentAllocator

import pynvml
pynvml.nvmlInit()

import os
import json

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

        self.allocator = AgentAllocator(available_lanes)
        self._lane_process: dict[Tuple[int, str], subprocess.Popen] = {
            lane: proc for lane, proc in zip(available_lanes, self.servers)
        }
        self._agent_to_lane: dict[str, Tuple[int, str]] = {}
        print(f"Inference Engine ready. Managing {len(available_lanes)} agent lanes.")


    def get_vllm_launch_command(self, gpu_id: int, port: int, model_name: str) -> list:
        """
        Returns the vLLM server launch command for a given GPU and port.
        """
        return [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--host", "localhost",
                "--port", str(port),
                "--tensor-parallel-size", "1", # Each server is self-contained
                # Removed --device call because we set it in the env["CUDA...""]
                "--model", model_name
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
                    command = self.get_vllm_launch_command(gpu_id, current_port, "facebook/opt-125m")

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
                command = self.get_vllm_launch_command(gpu_id, current_port, "facebook/opt-125m")

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

    def _server_has_model(self, lane_url: str, model_name: str) -> bool:
        try:
            resp = requests.get(f"{lane_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return any(m.get("id") == model_name for m in data.get("data", []))
        except requests.RequestException:
            pass
        return False

    def _load_model_on_server(self, lane_url: str, model_name: str):
        """Issue POST to load model if not present."""
        if self._server_has_model(lane_url, model_name):
            return
        try:
            resp = requests.post(f"{lane_url}/v1/models", json={"name": model_name}, timeout=10)
            if resp.status_code != 200:
                print(f"[Engine] Failed to load model {model_name} on {lane_url}: {resp.text}")
        except requests.RequestException as e:
            print(f"[Engine] Error contacting {lane_url}: {e}")

    def _unload_model_on_server(self, lane_url: str, model_name: str):
        try:
            requests.delete(f"{lane_url}/v1/models/{model_name}", timeout=5)
        except requests.RequestException:
            pass

    def preload_models(self, model_list: List[str]):
        """Synchronously ensure every model is loaded on some lane."""
        unique_models = list(dict.fromkeys(model_list))  # preserve order, uniq
        for m in unique_models:
            lane = self.allocator.acquire(f"__preload_{m}")
            self._load_model_on_server(lane[1], m)
            self.allocator.release(f"__preload_{m}")
        print(f"[Engine] Preloaded {len(unique_models)} models.")

    def register_agent(self, agent_id: str, model_name: str):
        """
        Assigns an agent to a GPU lane hosting *model_name*. If the model is
        not loaded yet, the engine will trigger a one-time load on that lane.
        (Implementation to follow based on docs)
        """
        lane = self.allocator.acquire(agent_id)
        self._agent_to_lane[agent_id] = lane
        print(f"Agent '{agent_id}' will use model '{model_name}' on GPU {lane[0]} ({lane[1]}).")

        # Restart process if current model differs
        if not self._server_has_model(lane[1], model_name):
            print(f"[Engine] Restarting vLLM server on {lane[1]} with model {model_name}.")
            proc = self._lane_process[lane]
            proc.terminate(); proc.wait()

            gpu_id = lane[0]
            port = int(lane[1].split(":")[-1])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            command = self.get_vllm_launch_command(gpu_id, port, model_name)
            new_proc = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self._lane_process[lane] = new_proc
            time.sleep(5)  # give it time to warm up

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

    def release_agent(self, agent_id: str):
        """Call when an agent no longer needs inference (game finished)."""
        lane = self._agent_to_lane.pop(agent_id, None)
        if lane:
            # terminate server process and restart empty for future use
            proc = self._lane_process[lane]
            proc.terminate(); proc.wait()
            gpu_id = lane[0]
            port = int(lane[1].split(":")[-1])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            # start blank server with dummy model for idle state
            command = self.get_vllm_launch_command(gpu_id, port, "facebook/opt-125m")
            new_proc = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self._lane_process[lane] = new_proc
            self.allocator.release(agent_id)
