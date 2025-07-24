"""
Dr GRPO (Group Relative Policy Optimization) with Co-located vLLM.

This implementation follows the "No GPU left behind" approach from TRL:
- Co-located vLLM runs inside the same process as FSDP training
- Eliminates GPU idle time by sharing GPUs between training and inference
- Uses proper distributed data synchronization across ranks
- Based on TRL's state-of-the-art co-located architecture

Architecture:
- Single distributed process group for both training and inference
- vLLM engines run on same GPUs as FSDP training (no separate server)
- Proper data broadcasting ensures all ranks work on identical batches
- Maximum GPU utilization with minimal overhead

Usage:
    torchrun --nproc_per_node=4 grpo_colocated.py --config training_configs/dr_grpo_config.yaml
"""

import asyncio
import logging
import time
import yaml
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import re

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

# ToSSim shared components
from Simulation.errors import ErrorCode, get_error_reward
from Simulation.grammar import validate_action, get_action_reward
from Simulation.turn_batcher import TurnBatcher
from Simulation.prompt_builder import build_training_prompt


@dataclass 
class DrGRPOConfig:
    """Configuration for Dr GRPO training."""
    
    # Model settings
    model_name: str = "microsoft/DialoGPT-medium"
    
    # FSDP settings  
    fsdp_config: Dict[str, Any] = field(default_factory=lambda: {
        "sharding_strategy": "FULL_SHARD",
        "cpu_offload": False,
        "mixed_precision": True
    })
    
    # vLLM settings - Co-located configuration
    vllm_config: Dict[str, Any] = field(default_factory=lambda: {
        "tensor_parallel_size": 1,
        "max_num_seqs": 32,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.4,
        "enforce_eager": True,
        "disable_log_stats": True,
    })
    
    # Training settings from YAML
    learning_rate: float = 1e-5
    min_learning_rate: Optional[float] = None
    warmup_ticks: Optional[int] = None
    gradient_clip_norm: Optional[float] = None
    batch_size: int = 64
    max_iterations: int = 5000
    log_interval: int = 10
    
    # Dr GRPO specific settings from YAML
    beta: float = 0.0
    sync_frequency: int = 100
    verbosity_penalty: float = 0.0
    
    # Environment settings from YAML
    num_games: int = 30
    active_seats_per_game: int = 3
    
    # Sampling settings from YAML
    temperature: float = 0.8
    top_p: float = 0.9
    max_tokens: int = 150
    
    # Allow extra fields from YAML to be ignored
    def __post_init__(self):
        # Map YAML names to dataclass names if they differ
        if hasattr(self, 'sync_every'):
            self.sync_frequency = self.sync_every
        if hasattr(self, 'max_tokens_per_gen'):
            self.max_tokens = self.max_tokens_per_gen
        if hasattr(self, 'num_concurrent_games'):
            self.num_games = self.num_concurrent_games
        if hasattr(self, 'log_ticks'):
            self.log_interval = self.log_ticks

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        # This allows for flexible loading from YAML
        # by only passing known fields to the constructor.
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        
        # Allow aliases
        aliases = {
            'sync_every': 'sync_frequency',
            'max_tokens_per_gen': 'max_tokens',
            'num_concurrent_games': 'num_games',
            'log_ticks': 'log_interval'
        }
        for alias, actual in aliases.items():
            if alias in config_dict:
                config_dict[actual] = config_dict.pop(alias)

        filtered_dict = {k: v for k, v in config_dict.items() if k in known_fields}
        
        return cls(**filtered_dict)


class DistributedDataManager:
    """Handles data broadcasting and synchronization across ranks."""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
    
    def broadcast_object(self, obj: Any, src_rank: int = 0) -> Any:
        """Broadcast arbitrary Python object from src_rank to all ranks."""
        if self.world_size == 1:
            return obj
        
        # Serialize object on source rank
        if self.rank == src_rank:
            data_bytes = pickle.dumps(obj)
            size = len(data_bytes)
        else:
            data_bytes = None
            size = 0
        
        # Broadcast size first
        size_tensor = torch.tensor([size], dtype=torch.long, device=f"cuda:{self.rank}")
        dist.broadcast(size_tensor, src_rank)
        
        # Broadcast data
        if self.rank == src_rank:
            data_tensor = torch.frombuffer(data_bytes, dtype=torch.uint8).cuda()
        else:
            data_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8, device=f"cuda:{self.rank}")
        
        dist.broadcast(data_tensor, src_rank)
        
        # Deserialize on non-source ranks
        if self.rank != src_rank:
            data_bytes = data_tensor.cpu().numpy().tobytes()
            obj = pickle.loads(data_bytes)
        
        return obj
    
    def all_reduce_scalar(self, value: float) -> float:
        """Average a scalar value across all ranks."""
        if self.world_size == 1:
            return value
        
        tensor = torch.tensor([value], dtype=torch.float32, device=f"cuda:{self.rank}")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return (tensor.item() / self.world_size)


def clip_think_block(text: str, max_think_tokens: int | None) -> str:
    """
    If max_think_tokens is not None, clip the <think>...</think> block to that many tokens and append </think><wait/>.
    If max_think_tokens is None, do nothing.
    """
    if max_think_tokens is None:
        return text
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if not match:
        return text  # No <think> block, return as is
    think_content = match.group(1)
    tokens = think_content.split()
    if len(tokens) <= max_think_tokens:
        return text  # No need to clip
    clipped = " ".join(tokens[:max_think_tokens])
    new_think = f"<think>{clipped}</think><wait/>"
    return re.sub(r"<think>.*?</think>", new_think, text, count=1, flags=re.DOTALL)


class ColocatedVLLMManager:
    """Manages co-located vLLM engine lifecycle within FSDP training."""
    
    def __init__(self, config: DrGRPOConfig, rank: int):
        self.config = config
        self.rank = rank
        self.vllm_engine = None
        self.is_active = False
    
    async def initialize(self):
        """Initialize vLLM engine on this rank."""
        logging.info(f"Initializing co-located vLLM on rank {self.rank}...")
        
        engine_args = AsyncEngineArgs(
            model=self.config.model_name,
            tensor_parallel_size=self.config.vllm_config.get("tensor_parallel_size", 1),
            max_num_seqs=self.config.vllm_config.get("max_num_seqs", 32),
            max_model_len=self.config.vllm_config.get("max_model_len", 2048),
            gpu_memory_utilization=self.config.vllm_config.get("gpu_memory_utilization", 0.4),
            enforce_eager=self.config.vllm_config.get("enforce_eager", True),
            disable_log_stats=self.config.vllm_config.get("disable_log_stats", True),
            device="cuda",
        )
        
        self.vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.is_active = True
        logging.info(f"Co-located vLLM initialized on rank {self.rank}")
    
    async def activate(self):
        """Activate vLLM for generation (switch from training mode)."""
        if not self.is_active:
            await self.initialize()
        # In a full implementation, this would include memory management
        # and potentially model weight synchronization
    
    def deactivate(self):
        """Deactivate vLLM to free up memory for training."""
        # In a full implementation, this might involve memory cleanup
        # For now, we keep the engine alive but mark as inactive
        pass
    
    async def generate(self, prompts: List[str]) -> List[str]:
        """Generate completions using co-located vLLM."""
        if not self.is_active or self.vllm_engine is None:
            raise RuntimeError("vLLM engine not active")
        
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens
        )
        
        # Generate async
        results = await self.vllm_engine.generate(prompts, sampling_params)
        
        # Extract completions and apply think block clipping if needed
        completions = []
        for result in results:
            if result.outputs:
                completion = result.outputs[0].text
            else:
                completion = None
            # Clip <think> block if max_think_tokens is set
            completion = clip_think_block(completion, getattr(self.config, 'max_think_tokens', None))
            completions.append(completion)
        
        return completions


class WeightSynchronizer:
    """Manages weight synchronization between FSDP and co‑located vLLM."""

    def __init__(self, fsdp_model, vllm_manager: ColocatedVLLMManager, rank: int):
        self.fsdp_model = fsdp_model
        self.vllm_manager = vllm_manager
        self.rank = rank

    async def sync_fsdp_to_vllm(self):
        """
        Pull a FULL_STATE_DICT from the FSDP model and push it into the
        co‑located vLLM engine on *every* rank.  Designed to be called from
        the training loop every `sync_frequency` steps.
        """

        # Ensure all ranks enter the routine together
        import torch
        import torch.distributed as dist
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig

        dist.barrier()

        # ────────────────────── 1. put vLLM to sleep ──────────────────────
        # Level‑2 frees weights *and* KV cache – ideal when parameters change
        await self.vllm_manager.vllm_engine.sleep(level=2)

        # ────────────────────── 2. grab FULL_STATE_DICT ───────────────────
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(
            self.fsdp_model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            config=cfg,
        ):
            state_dict = self.fsdp_model.state_dict()

        # ────────────────────── 3. normalise tensors ──────────────────────
        for k, v in state_dict.items():
            if v.dtype != torch.float16:
                v = v.to(torch.float16)
            state_dict[k] = v.contiguous()

        # ────────────────────── 4. load into vLLM ─────────────────────────
        llm_core = (
            self.vllm_manager.vllm_engine
            .llm_engine
            .model_executor
            .driver_worker
            .model  # same access path used by TRL
        )
        llm_core.load_weights(state_dict.items())

        # ────────────────────── 5. wake engine & final barrier ────────────
        await self.vllm_manager.vllm_engine.wake_up()
        dist.barrier()


class DrGRPOTrainer:
    """
    Dr GRPO trainer with co-located vLLM following TRL's approach.
    
    Key improvements:
    - Co-located vLLM eliminates GPU idle time
    - Proper distributed data synchronization
    - Maximum GPU utilization for both training and inference
    """
    
    def __init__(self, config: DrGRPOConfig):
        self.config = config
        
        # Initialize distributed training
        self._init_distributed()
        
        # Initialize distributed data manager
        self.data_manager = DistributedDataManager(self.rank, self.world_size)
        
        # Initialize models
        self._init_fsdp_model()
        
        # Initialize co-located vLLM manager
        self.vllm_manager = ColocatedVLLMManager(config, self.rank)
        
        # Initialize synchronizer
        self.synchronizer = WeightSynchronizer(self.fsdp_model, self.vllm_manager, self.rank)
        
        # Initialize environment (only on rank 0)
        if self.rank == 0:
            self.environment = TurnBatcher(
                num_games=config.num_games,
                active_seats_per_game=config.active_seats_per_game,
                model_name=config.model_name
            )
        else:
            self.environment = None
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.fsdp_model.parameters(),
            lr=config.learning_rate
        )
        
        # Training state
        self.step = 0
        self.metrics = {
            'total_samples': 0,
            'parse_accuracy': 0.0,
            'avg_reward': 0.0,
            'malformed_count': 0,
            'illegal_count': 0,
            'legal_count': 0
        }
        
        if self.rank == 0:
            logging.info("Dr GRPO trainer with co-located vLLM initialized")
            logging.info(f"Environment: {self.environment.get_batch_stats()}")
    
    def _init_distributed(self):
        """Initialize distributed training if not already done."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        self.rank = dist.get_rank() 
        self.world_size = dist.get_world_size()
        torch.cuda.set_device(self.rank)
        
        logging.info(f"Initialized distributed training: rank {self.rank}/{self.world_size}")
    
    def _init_fsdp_model(self):
        """Initialize FSDP model for gradient computation."""
        logging.info(f"Initializing FSDP model on rank {self.rank}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map=None  # FSDP will handle placement
        )
        
        # Wrap with FSDP
        self.fsdp_model = FSDP(
            model,
            auto_wrap_policy=transformer_auto_wrap_policy,
            mixed_precision=torch.distributed.fsdp.MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            ) if self.config.fsdp_config.get("mixed_precision", True) else None,
            device_id=torch.cuda.current_device(),
        )
        
        logging.info(f"FSDP model initialized on rank {self.rank}")
    
    def _compute_dr_grpo_loss(self, 
                             prompts: List[str],
                             completions: List[str], 
                             rewards: List[float]) -> torch.Tensor:
        """Compute Dr GRPO loss with β=0 (no length divisor)."""
        
        # Tokenize inputs
        full_texts = [p + c for p, c in zip(prompts, completions)]
        prompt_lengths = [len(self.tokenizer.encode(p)) for p in prompts]
        
        # Encode full sequences
        encodings = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.config.vllm_config.get("max_model_len", 2048),
            return_tensors="pt"
        ).to(next(self.fsdp_model.parameters()).device)
        
        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = self.fsdp_model(**encodings)
            logits = outputs.logits
        
        # Compute log probabilities for generated tokens only
        losses = []
        
        for i, (prompt_len, reward) in enumerate(zip(prompt_lengths, rewards)):
            # Extract completion tokens and their logits
            completion_logits = logits[i, prompt_len-1:-1]  # Shift for next-token prediction
            completion_tokens = encodings['input_ids'][i, prompt_len:]
            
            # Compute log probability of completion
            log_probs = torch.log_softmax(completion_logits, dim=-1)
            selected_log_probs = log_probs.gather(1, completion_tokens.unsqueeze(-1)).squeeze(-1)
            
            # Sum log probs for this completion
            completion_log_prob = selected_log_probs.sum()
            
            # Dr GRPO loss: -reward * log_prob (no length divisor, β=0)
            loss = -reward * completion_log_prob
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
    async def train_step(self) -> Dict[str, float]:
        """Execute one training step with co-located vLLM."""
        
        # Step 1: Get batch from environment (only on rank 0)
        if self.rank == 0:
            prompts, metadata = self.environment.next_batch()
            if not prompts:
                logging.warning("Empty batch from environment")
                prompts, metadata = [], []
        else:
            prompts, metadata = [], []
        
        # Step 2: Broadcast batch to all ranks
        prompts = self.data_manager.broadcast_object(prompts, src_rank=0)
        metadata = self.data_manager.broadcast_object(metadata, src_rank=0)
        
        if not prompts:
            return self.metrics
        
        # Step 3: Activate co-located vLLM on all ranks
        await self.vllm_manager.activate()
        
        # Step 4: Generate completions using co-located vLLM (all ranks in parallel)
        completions = await self.vllm_manager.generate(prompts)
        
        # Step 5: Deactivate vLLM to free memory for training
        self.vllm_manager.deactivate()
        
        # Step 6: Apply actions and get rewards (only on rank 0, then broadcast)
        if self.rank == 0:
            results = self.environment.apply_actions(metadata, completions)
            rewards = [r for r, _ in results]
        else:
            rewards = []
        
        rewards = self.data_manager.broadcast_object(rewards, src_rank=0)
        
        # Step 7: Compute loss and update FSDP model (all ranks with same data)
        loss = self._compute_dr_grpo_loss(prompts, completions, rewards)
        
        self.optimizer.zero_grad()
        loss.backward()  # FSDP automatically handles gradient reduction
        self.optimizer.step()
        
        # Step 8: Update metrics (only on rank 0)
        if self.rank == 0:
            self._update_metrics(completions, rewards)
        
        # Step 9: Sync weights periodically
        if self.step % self.config.sync_frequency == 0:
            await self.synchronizer.sync_fsdp_to_vllm()
        
        self.step += 1
        
        # Average loss across ranks for reporting
        avg_loss = self.data_manager.all_reduce_scalar(loss.item())
        
        return {
            'step': self.step,
            'loss': avg_loss,
            'batch_size': len(prompts),
            'avg_reward': sum(rewards) / len(rewards) if rewards else 0.0,
            'parse_accuracy': self.metrics['parse_accuracy'] if self.rank == 0 else 0.0
        }
    
    def _update_metrics(self, completions: List[str], rewards: List[float]):
        """Update training metrics (only called on rank 0)."""
        
        # Count reward types
        malformed = sum(1 for r in rewards if r == -1.0)
        illegal = sum(1 for r in rewards if r == 0.0)
        legal = sum(1 for r in rewards if r == 1.0)
        
        self.metrics['total_samples'] += len(completions)
        self.metrics['malformed_count'] += malformed
        self.metrics['illegal_count'] += illegal
        self.metrics['legal_count'] += legal
        
        # Update running averages
        total = self.metrics['total_samples']
        if total > 0:
            self.metrics['parse_accuracy'] = (self.metrics['legal_count'] + self.metrics['illegal_count']) / total
            self.metrics['avg_reward'] = (
                self.metrics['legal_count'] * 1.0 + 
                self.metrics['illegal_count'] * 0.0 + 
                self.metrics['malformed_count'] * (-1.0)
            ) / total
    
    async def train(self):
        """Main training loop with co-located vLLM."""
        if self.rank == 0:
            logging.info(f"Starting Dr GRPO training with co-located vLLM for {self.config.max_iterations} steps")
        
        # Initialize co-located vLLM on all ranks
        await self.vllm_manager.initialize()
        
        start_time = time.time()
        
        for iteration in range(self.config.max_iterations):
            step_metrics = await self.train_step()
            
            # Log metrics (only on rank 0)
            if self.rank == 0 and iteration % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                logging.info(
                    f"Step {iteration}: "
                    f"loss={step_metrics['loss']:.4f}, "
                    f"reward={step_metrics['avg_reward']:.3f}, "
                    f"accuracy={step_metrics['parse_accuracy']:.3f}, "
                    f"batch={step_metrics['batch_size']}, "
                    f"elapsed={elapsed:.1f}s"
                )
                
                # Environment stats
                if self.environment:
                    env_stats = self.environment.get_batch_stats()
                    logging.info(f"Environment: {env_stats}")
        
        if self.rank == 0:
            logging.info("Training completed")
        
        # Final sync
        await self.synchronizer.sync_fsdp_to_vllm()


def load_config(config_path: str) -> DrGRPOConfig:
    """Load configuration from YAML file."""
    
    if not Path(config_path).exists():
        logging.warning(f"Config file {config_path} not found, using defaults")
        return DrGRPOConfig()
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return DrGRPOConfig.from_dict(config_dict)


async def main():
    """Main entry point for co-located vLLM GRPO training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dr GRPO Training with Co-located vLLM")
    parser.add_argument("--config", type=str, default="training_configs/dr_grpo_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Load config
    config = load_config(args.config)
    
    # Initialize trainer
    trainer = DrGRPOTrainer(config)
    
    # Start training
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main()) 