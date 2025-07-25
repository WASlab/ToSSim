"""
Dr GRPO (Group Relative Policy Optimization)
=================================================================================

This is an updated version of the GRPO training script.  It retains the original
features (support for fp16/bf16 and bitsandbytes 4‑bit/8‑bit quantization, DDP
and FSDP distributed training, optional vLLM inference, CSV logging, and
early stopping) while incorporating the following improvements:

* **FlashAttention‑2 support:** When `use_flash_attention` is set to
  `True` (the default), the model is loaded with
  `attn_implementation="flash_attention_2"` as recommended by the
  Hugging Face documentation【829383066269290†L285-L304】.  This enables a more memory‑
  efficient attention implementation that still works with bf16/fp16 or
  bitsandbytes‑quantized weights.  Make sure the model is moved to the GPU
  before using FlashAttention‑2【829383066269290†L303-L305】.

* **Liger‑Kernel patching:** When `use_liger` is enabled and the
  `liger_kernel` package is installed, the script automatically applies
  the appropriate Liger kernel patch for the loaded model (Gemma, LLaMA,
  Mistral or Mixtral).  Liger kernels fuse RMSNorm, RoPE, SwiGLU/GEGLU and
  CrossEntropy layers to reduce memory usage by up to 60 %【362111756373219†L385-L390】.

* **Robust distributed synchronization:** A helper `_sync_bool()` uses
  `all_reduce` to synchronize boolean flags across ranks, preventing race
  conditions that can lead to hangs.  All calls to `dist.barrier()` are
  guarded by `dist.is_initialized()` to avoid errors when the process
  group has already been destroyed.

* **Updated generation API:** The `synced_gpus` argument has been removed
  from `GenerationConfig` (which now follows the upstream API) and
  instead passed directly to `generate()`.  This fixes
  `ValueError: Argument 'synced_gpus' is not a valid argument of
  GenerationConfig` on newer transformers releases.

To train, prepare a YAML configuration as before, adding optional
`use_flash_attention` and `use_liger` fields if desired.
"""

from __future__ import annotations

import os

# Environment hints for NCCL; see PyTorch docs for details.
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_DEBUG", "INFO")

import math
import yaml
import csv
import json
import logging
import asyncio
import functools
import tempfile
import traceback
import shutil
import re
import contextlib
from pathlib import Path
from dataclasses import dataclass, field, fields as dc_fields
from typing import Any, Dict, List, Optional
from datetime import timedelta

from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.utils.import_utils import is_bitsandbytes_available
import os
if int(os.environ.get("CUDA_LAUNCH_BLOCKING", "0")):
    torch.cuda.synchronize()

try:
    from transformers import BitsAndBytesConfig  # HF >= 4.30
except Exception:
    BitsAndBytesConfig = None  # type: ignore

# Try to import Liger kernels.  If unavailable, leave `_HAS_LIGER` false.
try:
    from liger_kernel.transformers import (
        apply_liger_kernel_to_gemma,
        apply_liger_kernel_to_gemma2,
        apply_liger_kernel_to_gemma3_text,
        apply_liger_kernel_to_llama,
        apply_liger_kernel_to_mistral,
        apply_liger_kernel_to_mixtral,
    )
    _HAS_LIGER = True
except Exception:
    _HAS_LIGER = False

try:
    import bitsandbytes as bnb
    from bitsandbytes.optim import PagedAdamW8bit
    _HAS_BNB = True
except Exception:  # pragma: no cover
    bnb = None
    PagedAdamW8bit = None  # type: ignore
    _HAS_BNB = False

# vLLM (kept but automatically disabled for 4/8bit)
try:
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    _HAS_VLLM = True
except Exception:
    _HAS_VLLM = False

# ToSSim deps
from Simulation.turn_batcher import TurnBatcher  # type: ignore

###############################################################################
# Configurations
###############################################################################

def _dtype() -> torch.dtype:
    """Return bf16 if available, else fp16."""
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _build_mp(dtype: torch.dtype) -> MixedPrecision:
    """Build a MixedPrecision object for FSDP."""
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def _sharding(name: str) -> ShardingStrategy:
    """Map a string to a ShardingStrategy enum."""
    name = name.upper()
    return {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }[name]


@dataclass
class FSDPConfig:
    """Configuration for FSDP."""
    sharding_strategy: str = "FULL_SHARD"
    sync_state: str = "SHARDED"  # FULL | SHARDED for vLLM sync
    mixed_precision: bool = True
    activation_checkpointing: bool = True
    use_orig_params: bool = True
    offload_full_state_dict_to_cpu: bool = True
    rank0_only_state_dict: bool = True
    checkpoint_dir: str | None = "checkpoints"


@dataclass
class VLLMConfig:
    """Configuration for optional vLLM integration."""
    tensor_parallel_size: int = 1
    max_num_seqs: int = 32
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.4
    enforce_eager: bool = True
    disable_log_stats: bool = True


@dataclass
class QuantArgs:
    """Arguments for bitsandbytes quantization."""
    # Only used when quant_mode in {"4bit", "8bit"}
    bnb_4bit_compute_dtype: str = "bfloat16"  # "bfloat16" | "float16"
    bnb_4bit_quant_type: str = "nf4"  # nf4 (recommended) | fp4
    bnb_4bit_use_double_quant: bool = True
    llm_int8_threshold: float = 6.0
    llm_int8_has_fp16_weight: bool = False

    def to_bnb(self):
        """Convert quantization arguments to a BitsAndBytesConfig."""
        if BitsAndBytesConfig is None:
            raise RuntimeError("You need a recent transformers for BitsAndBytesConfig")
        # map strings to torch dtypes
        dmap = {"bfloat16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}
        compute_dtype = dmap.get(self.bnb_4bit_compute_dtype.lower(), torch.bfloat16)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            llm_int8_threshold=self.llm_int8_threshold,
            llm_int8_has_fp16_weight=self.llm_int8_has_fp16_weight,
        )


@dataclass
class DrGRPOConfig:
    """Top‑level configuration for the DrGRPO trainer."""
    # core
    model_name: str = "google/gemma-3-27b-it"
    learning_rate: float = 1e-5
    min_learning_rate: float | None = 1e-6
    warmup_ticks: int = 200
    gradient_clip_norm: float = 1.0
    batch_size: int = 64
    max_iterations: int = 5000
    log_interval: int = 10
    k_per_prompt: int = 4
    beta: float = 0.0
    sync_frequency: int = 100
    use_flash_attention: bool = True  # enable FlashAttention‑2
    use_liger: bool = False  # enable Liger kernel patching

    # visibility / samples
    sample_log_interval: int = 100
    sample_log_n: int = 2
    samples_path: str = "grpo_samples.jsonl"

    # env
    num_games: int = 30
    active_seats_per_game: int = 3

    # sampling
    temperature: float = 0.8
    top_p: float = 0.9
    max_tokens: int = 256
    max_think_tokens: int | None = None

    # early stop
    early_stop_parse_rate: float | None = None
    early_stop_consecutive_ticks: int = 0

    # switches
    disable_vllm: bool = False
    only_transformers: bool = False

    # quantization
    quant_mode: str = "bf16"  # bf16 | fp16 | 8bit | 4bit
    quant_args: QuantArgs = field(default_factory=QuantArgs)

    # logging / outputs
    csv_path: str = "training_log.csv"
    track_weight_deltas: bool = True

    # Hub export
    output_hub_repo: Optional[str] = None
    hub_private: bool = True
    max_shard_size: str = "10GB"

    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    vllm_config: VLLMConfig = field(default_factory=VLLMConfig)

    @classmethod
    def from_yaml(cls, p: str | Path):
        """Load a configuration from a YAML file."""
        raw = yaml.safe_load(Path(p).read_text())
        fsdp_raw = raw.pop("fsdp_config", {})
        vllm_raw = raw.pop("vllm_config", {})
        qa_raw = raw.pop("quant_args", {})
        valid = {f.name for f in dc_fields(cls)}
        main_kwargs = {k: v for k, v in raw.items() if k in valid}
        fsdp = FSDPConfig(**fsdp_raw)
        vllm = VLLMConfig(**vllm_raw)
        qa = QuantArgs(**qa_raw)
        return cls(**main_kwargs, fsdp_config=fsdp, vllm_config=vllm, quant_args=qa)


###############################################################################
# Helper functions
###############################################################################

def _gemma_layer(model: nn.Module):
    """Return the decoder layer class for Gemma models."""
    for m in model.modules():
        if m.__class__.__name__.endswith("DecoderLayer"):
            return m.__class__
    raise RuntimeError("Cannot find Gemma DecoderLayer class")


def _clip_think_block(text: str, max_think_tokens: int | None) -> str:
    """Optionally truncate the content inside <think>…</think> to a given token count."""
    if max_think_tokens is None or not text:
        return text
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if not m:
        return text
    toks = m.group(1).split()
    if len(toks) <= max_think_tokens:
        return text
    clipped = " ".join(toks[:max_think_tokens])
    new_think = f"<think>{clipped}</think><wait/>"
    return re.sub(r"<think>.*?</think>", new_think, text, count=1, flags=re.DOTALL)


###############################################################################
# vLLM Manager
###############################################################################

class VLLMManager:
    """Asynchronous interface to a vLLM engine."""

    def __init__(self, cfg: DrGRPOConfig, rank: int):
        self.cfg = cfg
        self.rank = rank
        self.engine: "AsyncLLMEngine | None" = None

    async def init(self):
        """Initialize the underlying vLLM engine."""
        if self.engine:
            return
        if not _HAS_VLLM:
            raise RuntimeError("vLLM not installed but disable_vllm=False")
        v = self.cfg.vllm_config
        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=self.cfg.model_name,
                tensor_parallel_size=v.tensor_parallel_size,
                max_num_seqs=v.max_num_seqs,
                max_model_len=v.max_model_len,
                gpu_memory_utilization=v.gpu_memory_utilization,
                enforce_eager=v.enforce_eager,
                disable_log_stats=v.disable_log_stats,
            )
        )

    async def generate(self, prompts: List[str]) -> List[str]:
        """Generate completions for a list of prompts using vLLM."""
        sp = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_tokens,
        )
        res = await self.engine.generate(prompts, sp)
        outs = []
        for r in res:
            t = r.outputs[0].text if r.outputs else ""
            t = _clip_think_block(t, self.cfg.max_think_tokens)
            outs.append(t)
        return outs

    async def sleep(self):
        """Put the vLLM engine to sleep (release GPU resources)."""
        if self.engine:
            await self.engine.sleep(level=2)

    async def wake(self):
        """Wake the vLLM engine."""
        if self.engine:
            await self.engine.wake_up()

    def load_weights(self, sd: Dict[str, torch.Tensor]):
        """Load weights into the vLLM engine from a state dict."""
        core = (self.engine.llm_engine.model_executor.driver_worker.model)  # type: ignore
        core.load_weights(sd.items())


###############################################################################
# CSV Logger
###############################################################################

class CSVLogger:
    """Simple CSV logger that writes only on rank 0."""

    def __init__(self, path: str, rank: int):
        self.rank = rank
        if rank == 0:
            self.f = open(path, "w", newline="")
            self.w = csv.writer(self.f)
            self.w.writerow(["step", "loss", "avg_reward", "parse_acc", "weight_delta"])
            self.f.flush()
        else:
            self.f = None
            self.w = None

    def log(self, row: List[Any]):
        if self.rank == 0:
            self.w.writerow(row)
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


###############################################################################
# Trainer
###############################################################################

class Trainer:
    """Main GRPO training loop."""

    def __init__(self, cfg: DrGRPOConfig):
        self.cfg = cfg
        self._init_dist()

        # auto turn off vLLM if quantized
        if cfg.quant_mode.lower() in {"4bit", "8bit"}:
            if not cfg.only_transformers:
                logging.warning(
                    "quant_mode is %s → forcing disable_vllm=True (vLLM can't use bnb) and suggesting only_transformers=True",
                    cfg.quant_mode,
                )
            self.cfg.disable_vllm = True

        if cfg.only_transformers:
            logging.warning(
                "only_transformers=True → using pure HF + DDP (no FSDP, no vLLM)."
            )
            self.cfg.disable_vllm = True  # force
        elif cfg.disable_vllm:
            logging.warning(
                "disable_vllm=True → running FSDP with HF.generate (no vLLM)."
            )

        self.logger = CSVLogger(cfg.csv_path, self.rank)
        self.env: Optional[TurnBatcher] = (
            TurnBatcher(cfg.num_games, cfg.active_seats_per_game, cfg.model_name)
            if self.rank == 0
            else None
        )

        # Build model (FSDP or DDP) with quantization
        self._init_model()

        # vLLM only if allowed
        self.vllm = (
            None
            if (
                cfg.disable_vllm
                or cfg.only_transformers
                or cfg.quant_mode.lower() in {"4bit", "8bit"}
            )
            else VLLMManager(cfg, self.rank)
        )

        self._maybe_track_init_weights()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._build_sched()
        self.step = 0

    # -----------------------------------------------------------------------
    # Distributed initialization
    # -----------------------------------------------------------------------
    def _init_dist(self) -> None:
        """Initialize torch.distributed if necessary and set the device."""
        if not dist.is_initialized():
            # Long timeout to avoid watchdog killing long generate() steps
            dist.init_process_group("nccl", timeout=timedelta(hours=6))
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        torch.cuda.set_device(self.rank)

    # -----------------------------------------------------------------------
    # Model initialization
    # -----------------------------------------------------------------------
    def _quant_config(self):
        """Return quantization configuration or None for full precision."""
        m = self.cfg.quant_mode.lower()
        if m in {"bf16", "fp16"}:
            return None
        if not is_bitsandbytes_available() or not _HAS_BNB:
            raise RuntimeError("bitsandbytes not installed but quant_mode requires it")
        if m == "8bit":
            return "8bit"
        if m == "4bit":
            return self.cfg.quant_args.to_bnb()
        raise ValueError(f"Unknown quant_mode: {self.cfg.quant_mode}")

    def _init_model(self) -> None:
        """Load the model and wrap it for distributed training."""
        # Load tokenizer and set pad token
        self.tok = AutoTokenizer.from_pretrained(self.cfg.model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        # Build kwargs for model loading based on quantization
        quant = self._quant_config()
        load_kwargs: Dict[str, Any] = {}
        if quant is None:
            # native bf16/fp16
            load_kwargs.update(dict(torch_dtype=_dtype(), low_cpu_mem_usage=True))
        elif quant == "8bit":
            load_kwargs.update(dict(load_in_8bit=True, device_map={"": self.rank}))
        else:  # BitsAndBytesConfig 4bit
            load_kwargs.update(dict(quantization_config=quant, device_map={"": self.rank}))
        # Enable FlashAttention‑2 if requested
        if self.cfg.use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        # Optionally patch the model for Liger kernels
        if self.cfg.use_liger and _HAS_LIGER:
            try:
                name = self.cfg.model_name.lower()
                if "gemma-3" in name:
                    apply_liger_kernel_to_gemma3_text()
                elif "gemma2" in name:
                    apply_liger_kernel_to_gemma2()
                elif "gemma" in name:
                    apply_liger_kernel_to_gemma()
                elif "llama" in name:
                    apply_liger_kernel_to_llama()
                elif "mistral" in name:
                    apply_liger_kernel_to_mistral()
                elif "mixtral" in name:
                    apply_liger_kernel_to_mixtral()
            except Exception as e:
                logging.warning(f"Liger kernel patch failed: {e}")
        elif self.cfg.use_liger and not _HAS_LIGER:
            logging.warning(
                "use_liger=True but liger_kernel is not installed; skipping Liger patch."
            )

        # Load the model
        base = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **load_kwargs)
        device = torch.device(f"cuda:{self.rank}")
        quantized = self.cfg.quant_mode.lower() in {"4bit", "8bit"}

        # DDP path for quantized or only_transformers
        if self.cfg.only_transformers or quantized:
            # Ensure model is on the correct device for FlashAttention‑2
            base.to(device)
            self.ddp = DDP(
                base,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False,
            )
            self.fsdp = None
            self.net = self.ddp  # forward path
            self.core = self.ddp.module  # raw model
        else:
            # FSDP path for full‑precision training
            if self.cfg.fsdp_config.activation_checkpointing:
                apply_activation_checkpointing(
                    base,
                    checkpoint_wrapper_fn=functools.partial(
                        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                    ),
                    check_fn=lambda m: m.__class__.__name__.endswith("DecoderLayer"),
                )
            awp = functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls={_gemma_layer(base)}
            )
            self.fsdp = FSDP(
                base,
                auto_wrap_policy=awp,
                sharding_strategy=_sharding(self.cfg.fsdp_config.sharding_strategy),
                mixed_precision=_build_mp(_dtype())
                if self.cfg.fsdp_config.mixed_precision
                else None,
                device_id=torch.cuda.current_device(),
                use_orig_params=self.cfg.fsdp_config.use_orig_params,
            )
            self.ddp = None
            self.net = self.fsdp
            self.core = self.fsdp.module

    # -----------------------------------------------------------------------
    # Optimizer and scheduler
    # -----------------------------------------------------------------------
    def params(self):
        return self.net.parameters()

    def _init_optimizer(self):
        """Initialize the optimizer."""
        if (
            self.cfg.quant_mode.lower() in {"4bit", "8bit"}
            and _HAS_BNB
            and PagedAdamW8bit is not None
        ):
            logging.info("Using bitsandbytes PagedAdamW8bit optimizer")
            return PagedAdamW8bit(self.params(), lr=self.cfg.learning_rate)
        else:
            return torch.optim.AdamW(self.params(), lr=self.cfg.learning_rate)

    def _build_sched(self):
        """Create a cosine schedule with warmup."""
        warm, total, minlr = (
            self.cfg.warmup_ticks,
            self.cfg.max_iterations,
            (self.cfg.min_learning_rate or 0.0),
        )

        def decay(i):
            if i < warm:
                return i / max(1, warm)
            prog = (i - warm) / max(1, total - warm)
            return max(minlr / self.cfg.learning_rate, 0.5 * (1 + math.cos(math.pi * prog)))

        return LambdaLR(self.optimizer, decay)

    # -----------------------------------------------------------------------
    # Synchronization helpers
    # -----------------------------------------------------------------------
    def _sync_bool(self, flag: bool) -> bool:
        """Synchronize a boolean flag across all ranks using all_reduce."""
        t = torch.tensor([1 if flag else 0], dtype=torch.int, device=f"cuda:{self.rank}")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.item() > 0

    # -----------------------------------------------------------------------
    # Weight deltas for monitoring
    # -----------------------------------------------------------------------
    def _maybe_track_init_weights(self) -> None:
        self.track = self.cfg.track_weight_deltas and (self.rank == 0)
        if self.track:
            self.init_vec = torch.nn.utils.parameters_to_vector(
                [p.detach().cpu() for p in self.params()]
            )
        else:
            self.init_vec = None

    def _current_delta(self) -> float:
        if not self.track:
            return 0.0
        cur = torch.nn.utils.parameters_to_vector(
            [p.detach().cpu() for p in self.params()]
        )
        return torch.norm(cur - self.init_vec).item()

    # -----------------------------------------------------------------------
    # Save to Hugging Face Hub
    # -----------------------------------------------------------------------
    def _push_to_hub(self) -> None:
        """Save and upload the model to the Hugging Face Hub."""
        if self.cfg.output_hub_repo is None:
            if self.rank == 0:
                logging.info("output_hub_repo not set — skipping push_to_hub.")
            return
        # Synchronize before exporting
        if dist.is_initialized():
            dist.barrier()
        if self.rank != 0:
            if dist.is_initialized():
                dist.barrier()
            return
        logging.info("Exporting state_dict for push_to_hub …")
        tmp = tempfile.mkdtemp()
        try:
            if self.fsdp is None:
                # DDP / quantized
                state_dict = self.core.state_dict()
                self.core.save_pretrained(
                    tmp,
                    state_dict=state_dict,
                    safe_serialization=True,
                    max_shard_size=self.cfg.max_shard_size,
                )
            else:
                # FSDP full state
                fsd_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(
                    self.fsdp, StateDictType.FULL_STATE_DICT, fsd_cfg
                ):
                    state_dict = self.fsdp.state_dict()
                self.core.save_pretrained(
                    tmp,
                    state_dict=state_dict,
                    safe_serialization=True,
                    max_shard_size=self.cfg.max_shard_size,
                )
            self.tok.save_pretrained(tmp)
            logging.info(f"Pushing to Hub: {self.cfg.output_hub_repo}")
            from huggingface_hub import create_repo, upload_folder

            create_repo(
                self.cfg.output_hub_repo,
                exist_ok=True,
                repo_type="model",
                private=self.cfg.hub_private,
            )
            upload_folder(
                repo_id=self.cfg.output_hub_repo,
                folder_path=tmp,
                repo_type="model",
            )
            logging.info("Push to Hub complete.")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if dist.is_initialized():
            dist.barrier()

    # -----------------------------------------------------------------------
    # Sample logging
    # -----------------------------------------------------------------------
    def _maybe_log_samples(self, prompts: List[str], outs: List[str], rewards: List[float]) -> None:
        """Log sample completions to file and console on rank 0."""
        if self.rank != 0:
            return
        n = min(self.cfg.sample_log_n, len(outs))
        with open(self.cfg.samples_path, "a") as f:
            for i in range(n):
                rec = {
                    "step": self.step,
                    "prompt": prompts[i],
                    "completion": outs[i],
                    "reward": float(rewards[i]) if i < len(rewards) else None,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logging.info(
                    "\n[SAMPLE %d @ step %d]\nPrompt:\n%s\n---\nCompletion:\n%s\n---\nReward: %s\n",
                    i,
                    self.step,
                    prompts[i],
                    outs[i],
                    rec["reward"],
                )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    async def train(self) -> None:
        """Perform the training loop."""
        if self.vllm is not None:
            await self.vllm.init()
        consecutive_hits = 0
        total_steps = self.cfg.max_iterations
        try:
            for it in trange(total_steps, desc="Training", disable=(self.rank != 0)):
                self.step = it
                stats = await self._step()
                # Logging to stdout
                if self.rank == 0:
                    tqdm.write(
                        f"Step {it}: loss={stats['loss']:.4f}, avg_reward={stats['avg_reward']:.4f}, parse_acc={stats['parse_acc']:.4f}"
                    )
                    if it % self.cfg.log_interval == 0:
                        logging.info(json.dumps(stats))
                # Early stopping
                stop = False
                if (
                    self.rank == 0
                    and self.cfg.early_stop_parse_rate is not None
                    and self.cfg.early_stop_consecutive_ticks > 0
                ):
                    current_acc = stats.get("parse_acc", 0.0)
                    consecutive_hits = (
                        consecutive_hits + 1
                        if current_acc >= self.cfg.early_stop_parse_rate
                        else 0
                    )
                    if consecutive_hits >= self.cfg.early_stop_consecutive_ticks:
                        logging.info(
                            f"[early-stop] parse_acc {current_acc:.4f} for {consecutive_hits} consecutive steps. Stopping."
                        )
                        stop = True
                # Synchronize stop flag
                stop = self._sync_bool(stop)
                if stop:
                    break
        except Exception:
            # Save checkpoint on fatal error
            if self.rank == 0:
                logging.error("Fatal error, saving checkpoint…")
                ckdir = Path(self.cfg.fsdp_config.checkpoint_dir or "checkpoints")
                ckdir.mkdir(parents=True, exist_ok=True)
                try:
                    if self.fsdp is None:
                        torch.save(self.core.state_dict(), ckdir / "fatal_ckpt_ddp.pt")
                    else:
                        fsd_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                        with FSDP.state_dict_type(
                            self.fsdp, StateDictType.FULL_STATE_DICT, fsd_cfg
                        ):
                            torch.save(self.fsdp.state_dict(), ckdir / "fatal_ckpt_fsdp.pt")
                except Exception as e:
                    logging.error(f"Failed to save fatal checkpoint: {e}")
                logging.error(traceback.format_exc())
            # Bring all ranks down cleanly
            with contextlib.suppress(Exception):
                if dist.is_initialized():
                    dist.destroy_process_group()
            raise
        finally:
            if self.rank == 0:
                # Push model to hub and plot metrics if training finished normally
                self._push_to_hub()
                try:
                    import pandas as pd
                    import matplotlib.pyplot as plt

                    df = pd.read_csv(self.cfg.csv_path)
                    plt.plot(df["step"], df["avg_reward"], label="avg_reward")
                    plt.plot(df["step"], df["parse_acc"], label="parse_acc")
                    plt.legend()
                    plt.xlabel("step")
                    plt.savefig("training_metrics.png")
                    plt.close()
                except Exception as e:
                    logging.warning(f"Plotting failed: {e}")
            self.logger.close()
            # Clean up process group
            if dist.is_initialized():
                with contextlib.suppress(Exception):
                    dist.barrier()
                with contextlib.suppress(Exception):
                    dist.destroy_process_group()

    # -----------------------------------------------------------------------
    # Single training step
    # -----------------------------------------------------------------------
    async def _step(self):
        """Perform a single training step and return logging stats."""
        # Get batch on rank 0 and broadcast to others
        if self.rank == 0:
            prompts, meta = self.env.next_batch()
        else:
            prompts, meta = [], []
        prompts = self._bcast(prompts)
        meta = self._bcast(meta)
        if not prompts:
            return {
                "step": self.step,
                "loss": 0.0,
                "avg_reward": 0.0,
                "parse_acc": 0.0,
                "weight_delta": 0.0,
            }
        # k sampling
        K = self.cfg.k_per_prompt
        tp = [p for p in prompts for _ in range(K)]
        group_ids = torch.arange(len(prompts), device=f"cuda:{self.rank}").repeat_interleave(K)
        # Generation
        if self.vllm is None:
            outs = self._generate_with_hf(tp)
        else:
            outs = await self.vllm.generate(tp)
        # Rewards
        if self.rank == 0:
            rewards = [r for r, _ in self.env.apply_actions(meta * K, outs)]
        else:
            rewards = []
        rewards = self._bcast(rewards)
        # Sample log
        if self.rank == 0 and self.step % self.cfg.sample_log_interval == 0:
            self._maybe_log_samples(tp, outs, rewards)
        # Loss
        lp = self._logprob_sums(tp, outs)
        loss = self._grpo_loss(lp, torch.tensor(rewards, device=lp.device), group_ids)
        # Optimizer step
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params(), self.cfg.gradient_clip_norm)
        self.optimizer.step()
        self.scheduler.step()
        # Sync weights if vLLM is alive
        if self.vllm is not None and (self.step % self.cfg.sync_frequency == 0):
            await self._sync_weights()
        # Logging values
        loss_val = self._mean(loss.item())
        rew_avg = sum(rewards) / max(1, len(rewards))
        delta = self._current_delta()
        parse_acc = 0.0
        if self.rank == 0:
            parse_acc = self.env.get_batch_stats()["parsing_accuracy"]
        self.logger.log([self.step, loss_val, rew_avg, parse_acc, delta])
        return {
            "step": self.step,
            "loss": loss_val,
            "avg_reward": rew_avg,
            "parse_acc": parse_acc,
            "weight_delta": delta,
        }

    # -----------------------------------------------------------------------
    # HF-only generation path
    # -----------------------------------------------------------------------
    def _generate_with_hf(self, prompts: List[str]) -> List[str]:
        if self.rank != 0:
            if dist.is_initialized():
                dist.barrier()
            return self._bcast_obj([], src=0)

        self.net.eval()
        gen_cfg = GenerationConfig(
            do_sample=True,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_new_tokens=self.cfg.max_tokens,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )

        outs: List[str] = []
        device = next(self.params()).device
        with torch.no_grad():
            for p in prompts:
                try:
                    enc = self.tok(p, return_tensors="pt").to(device)
                    out_ids = self.core.generate(
                        **enc, generation_config=gen_cfg, synced_gpus=False
                    )
                except RuntimeError as e:
                    # Catch the device-side assert early and recover gracefully
                    logging.error(f"[rank {self.rank}] generate() failed: {e}")
                    logging.error("Retrying with greedy decoding to skip NaN probs.")
                    enc = self.tok(p, return_tensors="pt").to(device)
                    out_ids = self.core.generate(
                        **enc,
                        do_sample=False,
                        max_new_tokens=self.cfg.max_tokens,
                        pad_token_id=self.tok.pad_token_id,
                        eos_token_id=self.tok.eos_token_id,
                        synced_gpus=False,
                    )

                text = self.tok.decode(
                    out_ids[0][enc["input_ids"].shape[1]:],
                    skip_special_tokens=False
                )
                text = _clip_think_block(text, self.cfg.max_think_tokens)
                outs.append(text)

        self.net.train()
        if dist.is_initialized():
            dist.barrier()
        return self._bcast_obj(outs, src=0)


    # -----------------------------------------------------------------------
    # Weight synchronization for vLLM
    # -----------------------------------------------------------------------
    async def _sync_weights(self) -> None:
        """Synchronize weights from FSDP to vLLM."""
        mode = self.cfg.fsdp_config.sync_state.upper()
        await self.vllm.sleep()
        if dist.is_initialized():
            dist.barrier()
        if mode == "FULL":
            cfg = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=self.cfg.fsdp_config.rank0_only_state_dict,
            )
            with FSDP.state_dict_type(
                self.fsdp, StateDictType.FULL_STATE_DICT, cfg
            ):
                sd = self.fsdp.state_dict()
            target = sd
        else:
            cfg = ShardedStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(
                self.fsdp, StateDictType.SHARDED_STATE_DICT, cfg
            ):
                sd = self.fsdp.state_dict()
            target = {k: v for k, v in sd.items()}
        self.vllm.load_weights(target)
        await self.vllm.wake()
        if dist.is_initialized():
            dist.barrier()

    # -----------------------------------------------------------------------
    # Broadcast helpers
    # -----------------------------------------------------------------------
    def _bcast(self, obj: Any, src: int = 0) -> Any:
        buf = [obj] if self.rank == src else [None]
        if dist.is_initialized():
            dist.broadcast_object_list(buf, src=src)
        return buf[0]

    def _bcast_obj(self, obj: Any, src: int = 0) -> Any:
        buf = [obj] if self.rank == src else [None]
        if dist.is_initialized():
            dist.broadcast_object_list(buf, src=src)
        return buf[0]


    def _mean(self, x: float) -> float:
        t = torch.tensor([x], device=f"cuda:{self.rank}")
        dist.all_reduce(t)
        return (t / self.world).item()

    def _logprob_sums(self, prompts: List[str], completions: List[str]) -> torch.Tensor:
        device = next(self.params()).device
        maxlen = self.cfg.vllm_config.max_model_len

        texts = [p + c for p, c in zip(prompts, completions)]
        plens = [len(self.tok.encode(p)) for p in prompts]

        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=maxlen,
            return_tensors="pt",
        ).to(device)

        # Bail out early if truncation ate the completion
        seq_len = enc["input_ids"].shape[1]
        for i, pl in enumerate(plens):
            if pl >= seq_len:
                raise RuntimeError(
                    f"Prompt length ({pl}) >= encoded length ({seq_len}). "
                    "Increase vllm_config.max_model_len or shorten prompts/completions."
                )

        use_amp = not (self.cfg.quant_mode.lower() in {"4bit", "8bit"})
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = self.net(**enc).logits

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise RuntimeError(f"NaN/Inf in logits at step {self.step} during _logprob_sums")

        res = []
        for i, pl in enumerate(plens):
            start = max(pl - 1, 0)
            lp_slice = logits[i, start:-1].float()  # use float32 for stability
            ids_slice = enc["input_ids"][i, pl:]

            if ids_slice.numel() == 0:
                res.append(torch.tensor(0.0, device=device, dtype=logits.dtype))
                continue

            lp = torch.log_softmax(lp_slice, dim=-1)
            if torch.isnan(lp).any() or torch.isinf(lp).any():
                lp = torch.nan_to_num(lp, nan=-1e9, posinf=-1e9, neginf=-1e9)

            token_ll = lp.gather(1, ids_slice.unsqueeze(-1)).squeeze(-1)
            res.append(token_ll.sum())

        return torch.stack(res)


    def _grpo_loss(self, lp: torch.Tensor, rw: torch.Tensor, gid: torch.Tensor) -> torch.Tensor:
        adv = torch.zeros_like(rw)
        for g in torch.unique(gid):
            m = gid == g
            adv[m] = rw[m] - rw[m].mean()
        return -(adv * lp).mean()


###############################################################################
# Command‑line interface
###############################################################################

async def _amain() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = DrGRPOConfig.from_yaml(args.config)
    tr = Trainer(cfg)
    await tr.train()


if __name__ == "__main__":
    asyncio.run(_amain())