"""
The InferenceEngine is responsible for managing vLLM servers and orchestrating
requests to the LLM agents.
"""
import subprocess
import time
import requests
from typing import List, Tuple, Dict, Any, Optional, Generator
from .allocator import AgentAllocator
from .client import InferenceClient
import jinja2
from pathlib import Path

# NVML (NVIDIA Management Library) is only required when running the real
# inference engine on a machine with GPUs.  Many unit tests import this module
# without actually needing GPU support which would normally fail at import time
# if the NVML shared library is unavailable.  To allow these tests to run in
# CPU-only environments we attempt to import and initialise ``pynvml``
# defensively.
try:  # pragma: no cover - purely optional path
    import pynvml
    try:
        pynvml.nvmlInit()
        _NVML_AVAILABLE = True
    except Exception:
        # Library is present but cannot be initialised (e.g. no driver)
        _NVML_AVAILABLE = False
except Exception:  # pragma: no cover - optional dependency missing
    pynvml = None  # type: ignore
    _NVML_AVAILABLE = False

import os
import json
import re

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
            print(
                "Warning: No GPU lanes available. InferenceEngine will run in"
                " a disabled state."
            )

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
        if not _NVML_AVAILABLE:
            print("pynvml not available; assuming 0 GPUs.")
            return []
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

        except (FileNotFoundError, subprocess.CalledProcessError, AttributeError, Exception) as e:
            # AttributeError can occur if pynvml is present but missing library functions
            print("Warning: Unable to query GPUs via pynvml (%s). Assuming 0 GPUs." % e)
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
        if _NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
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

    # ------------------------------------------------------------------
    # Streaming Chat Support
    # ------------------------------------------------------------------

    def stream_chat_with_tools(self, 
                              agent_id: str, 
                              system_prompt: str, 
                              user_prompt: str, 
                              initial_observation: Optional[str] = None,
                              model_type: str = "gemma",
                              game=None,
                              player=None,
                              **sampling_params) -> Generator[str, None, None]:
        """
        Stream chat with tool detection and real-time chat updates.
        
        This method:
        1. Detects XML tool tags in the stream
        2. Executes tools when detected
        3. Injects observations and new chat messages
        4. Preserves prefix caching by only checking for new messages after tool execution
        """
        lane = self._agent_to_lane.get(agent_id)
        if not lane:
            raise RuntimeError(f"Agent {agent_id} not registered")
        
        client = InferenceClient(lane[1], "model")  # model name will be set by vLLM
        
        # Build initial conversation using Jinja templates
        conversation = self._build_conversation(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            observation=initial_observation,
            model_type=model_type
        )
        
        # Track seen messages to avoid duplicates
        seen_message_timestamps = set()
        if game and player:
            # Initialize with current messages
            current_messages = game.chat.get_visible_messages(player)
            seen_message_timestamps = {msg.timestamp for msg in current_messages}
        
        final_response = ""
        
        while True:
            # Generate with current conversation
            try:
                stream_response = client.chat(
                    conversation,
                    stream=True,
                    **sampling_params
                )
                
                response_buffer = ""
                xml_detected = False
                
                # Stream and detect XML
                for line in stream_response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]  # Remove 'data: ' prefix
                            if data == '[DONE]':
                                break
                            
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and chunk['choices']:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content = delta['content']
                                        response_buffer += content
                                        
                                        # Check for XML tool tags
                                        if self._should_stop_for_xml(response_buffer):
                                            xml_detected = True
                                            break
                            except json.JSONDecodeError:
                                continue
                
                if xml_detected:
                    # Execute tool using the existing tool router
                    from .tool_router import apply_first_tool_call
                    patched_text, tool_result = apply_first_tool_call(response_buffer, game=game, player=player)
                    
                    # Add assistant's partial response (use patched text to ensure proper XML)
                    conversation.append({"role": "assistant", "content": patched_text})
                    
                    # Add observation (properly formatted for vLLM)
                    conversation.append({"role": "observation", "content": tool_result})
                    
                    # Check for new chat messages ONLY after tool execution
                    if game and player:
                        new_messages = self._get_new_chat_messages(game, player, seen_message_timestamps)
                        if new_messages:
                            for msg in new_messages:
                                conversation.append({
                                    "role": "user", 
                                    "content": f"{msg.sender.name}: {msg.message}"
                                })
                                seen_message_timestamps.add(msg.timestamp)
                    
                    final_response += response_buffer
                    yield response_buffer  # Yield the partial response
                    continue
                else:
                    final_response += response_buffer
                    yield response_buffer  # Yield the final response
                    break
                    
            except Exception as e:
                print(f"[Engine] Error in streaming chat: {e}")
                break
        
        return final_response

    def _should_stop_for_xml(self, text: str) -> bool:
        """Check if we should stop generation due to XML tool tags."""
        # Look for complete XML tags (opening and closing)
        # This is a simplified version - grammar.py handles the full parsing
        xml_pattern = r'<[^>]+>.*?</[^>]+>'
        return bool(re.search(xml_pattern, text))

    def _get_new_chat_messages(self, game, player, seen_timestamps: set) -> List:
        """Get new chat messages since last check."""
        current_messages = game.chat.get_visible_messages(player)
        new_messages = [msg for msg in current_messages if msg.timestamp not in seen_timestamps]
        return new_messages

    def chat_with_tools(self, 
                       agent_id: str, 
                       system_prompt: str, 
                       user_prompt: str, 
                       initial_observation: Optional[str] = None,
                       model_type: str = "gemma",
                       **sampling_params) -> str:
        """
        Non-streaming version for batched processing (GRPO, etc.).
        No real-time chat updates needed.
        """
        lane = self._agent_to_lane.get(agent_id)
        if not lane:
            raise RuntimeError(f"Agent {agent_id} not registered")
        
        client = InferenceClient(lane[1], "model")
        
        # Build conversation using Jinja templates
        conversation = self._build_conversation(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            observation=initial_observation,
            model_type=model_type
        )
        
        # Single generation
        response = client.chat(conversation, **sampling_params)
        return response['choices'][0]['message']['content']

    def _build_conversation(self, system_prompt: str, user_prompt: str, 
                           observation: Optional[str] = None, model_type: str = "gemma") -> List[Dict[str, str]]:
        """Build conversation using Jinja templates for proper vLLM compatibility."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if observation:
            messages.append({"role": "observation", "content": observation})
        
        return messages

    def _apply_chat_template(self, messages: List[Dict[str, str]], model_type: str = "gemma") -> str:
        """Apply the appropriate Jinja chat template for the model type."""
        template_dir = Path(__file__).parent / "templates"
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir), autoescape=False)
        
        if model_type.lower() == "gemma":
            template = env.get_template("gemma_chat_template.jinja")
        else:
            template = env.get_template("chat_template.jinja")
        
        return template.render(messages=messages)

    # ------------------------------------------------------------------
    # Usage Example
    # ------------------------------------------------------------------
    """
    Example usage in match_runner.py:
    
    # For streaming (live games with real-time chat updates)
    def process_agent_turn_streaming(self, game, player, agent_id):
        system_prompt = build_system_prompt(player.name, player.role, game)
        user_prompt = build_user_prompt(game, player)
        
        # Stream with tool detection and chat updates
        response_chunks = []
        for chunk in self.engine.stream_chat_with_tools(
            agent_id=agent_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_type="gemma",  # Use string, not enum
            game=game,
            player=player,
            temperature=0.7,
            max_tokens=1000
        ):
            response_chunks.append(chunk)
            # Optionally yield chunks for real-time display
        
        return ''.join(response_chunks)
    
    # For batched processing (GRPO, training)
    def process_agent_turn_batched(self, game, player, agent_id):
        system_prompt = build_system_prompt(player.name, player.role, game)
        user_prompt = build_user_prompt(game, player)
        
        # Single generation without real-time updates
        response = self.engine.chat_with_tools(
            agent_id=agent_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_type="gemma",  # Use string, not enum
            temperature=0.7,
            max_tokens=1000
        )
        
        return response
    """
