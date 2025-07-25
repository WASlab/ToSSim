"""
grpo.py
=============

This module implements Group‑Relative Policy Optimization (GRPO) and its
distributed version for online reinforcement learning with large language
models.  The code is adapted from ToSSim and incorporates a number of
improvements inspired by the HuggingFace TRL implementation of GRPO.

Key features
------------

* **Robust multinomial sampling:**  The default `torch.multinomial` will
  raise a device‑side assertion if given an all‑zero or otherwise invalid
  probability distribution.  The `safe_multinomial` wrapper defined here
  sanitises its input by clamping negative, NaN or infinite values to
  zero and replacing any zero‑sum row with a uniform distribution before
  sampling.  This function is installed globally as
  `torch.multinomial` to protect all sampling within the training loop.

* **Conservative generation defaults:**  Sampling from quantised models
  with aggressive `top_p`/`top_k` or minimum probability cutoffs can
  produce extremely sparse distributions, leading to NaNs or invalid
  distributions.  The `build_generation_config` helper constructs
  `GenerationConfig` objects with sensible defaults that disable
  nucleus and top‑k sampling unless explicitly requested.  Parameters
  such as `temperature`, `top_p`, `top_k`, `min_p` and
  `repetition_penalty` are exposed via the YAML configuration.

* **Fallback sampling:**  The `generate_with_retry` helper wraps
  `model.generate` and falls back to greedy decoding if an exception
  occurs during sampling.  This prevents occasional NaNs or invalid
  logits from aborting the entire training run.

* **Distributed training support:**  The `Trainer` class wraps the
  underlying model in either `DistributedDataParallel` or
  `FullyShardedDataParallel` depending on whether quantisation is used.
  Synchronisation utilities ensure that early stopping and fatal
  exceptions are propagated across all ranks.  Weight synchronisation
  to an optional vLLM engine is handled asynchronously.

* **Configurable loss:**  Two GRPO variants are supported via the
  `loss_type` field in the configuration: the original GRPO loss
  (normalising by the length of each completion) and the Dr.GRPO loss
  (dividing by a fixed maximum length).  Rewards can be optionally
  standardised per‑group (`scale_rewards`).

The code is intended to be fully self‑contained; it defines its own
dataclasses for configuration, helper functions for generation and
sampling and a high‑level training loop.  See the accompanying YAML
file for an example configuration.

"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import functools
import json
import logging
import math
import os
import re
import shutil
import tempfile
import traceback
from dataclasses import dataclass, field, fields as dc_fields
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils.import_utils import is_bitsandbytes_available

# Set environment hints for NCCL.  These improve robustness of
# multi‑GPU training and are recommended in PyTorch documentation.
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_DEBUG", "INFO")

# ----------------------------------------------------------------------------
# Safe multinomial and generation helpers
# ----------------------------------------------------------------------------

_orig_multinomial = torch.multinomial

def safe_multinomial(
    input: torch.Tensor,
    num_samples: int,
    replacement: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Robust wrapper around ``torch.multinomial``.

    This function cleans the input tensor by replacing NaNs and
    infinite values with zero and clamping negative values to zero,
    then samples from the resulting distribution.  If any row of the
    cleaned tensor sums to zero (meaning there is no mass to sample
    from), that row is replaced with a uniform distribution before
    sampling.  This prevents the ``invalid multinomial distribution``
    error that would otherwise occur when the sum of probabilities is
    not positive.

    Args:
        input: The input tensor of non‑negative weights.  May be 1D or
            higher dimensional; sampling is applied along the last
            dimension.
        num_samples: Number of samples to draw.
        replacement: Whether to sample with replacement.
        generator: Optional PRNG for deterministic sampling.

    Returns:
        A tensor of indices with shape ``input.shape[:-1] + (num_samples,)``.
    """
    # Convert to floating type if necessary
    if not torch.is_floating_point(input):
        input = input.float()
    # Replace NaNs and Infs with zeros and clamp negatives to zero
    clean = torch.nan_to_num(input, nan=0.0, posinf=0.0, neginf=0.0)
    clean = torch.clamp(clean, min=0.0)
    # Compute sums along the last dimension
    if clean.dim() == 1:
        total = clean.sum()
        if total <= 0:
            # Replace with uniform distribution to avoid invalid sampling
            clean = torch.ones_like(clean)
    else:
        # Identify rows whose sums are non‑positive
        sums = clean.sum(dim=-1, keepdim=True)
        mask = sums <= 0
        if mask.any():
            # Broadcast uniform distribution to those rows
            uniform = torch.ones_like(clean)
            clean = torch.where(mask, uniform, clean)
    return _orig_multinomial(clean, num_samples, replacement=replacement, generator=generator)

# Install the safe multinomial globally
torch.multinomial = safe_multinomial  # type: ignore

def build_generation_config(
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    max_new_tokens: int,
    pad_token_id: int,
    eos_token_id: int,
) -> GenerationConfig:
    """Construct a conservative ``GenerationConfig``.

    Parameters mirror those of ``transformers.GenerationConfig`` but
    provide sensible defaults that avoid pathological zero‑mass
    distributions.  ``top_k`` is interpreted as the maximum number of
    tokens to sample from; if ``None`` it is disabled (internally
    represented as ``-1`` by HuggingFace).  ``top_p`` is the nucleus
    sampling threshold; values of ``1.0`` disable nucleus sampling.

    Args:
        temperature: Softmax temperature.  Lower values encourage
            greedier sampling; values near zero effectively disable
            sampling.
        top_p: Top‑p (nucleus) sampling cutoff.  Set to 1.0 to disable.
        top_k: Top‑k sampling cutoff.  Set to ``None`` to disable.
        min_p: Minimum probability threshold relative to the maximum
            probability.  Tokens whose probability is less than
            ``min_p * max_prob`` are discarded.  Set to 0.0 to disable.
        repetition_penalty: Penalty factor applied to previously
            generated tokens; values >1 discourage repetition.
        max_new_tokens: Maximum number of tokens to generate.
        pad_token_id: Padding token identifier.
        eos_token_id: End‑of‑sequence token identifier.

    Returns:
        A ``GenerationConfig`` instance populated with the provided
        parameters.
    """
    # Translate None to sentinel value accepted by HF (‑1 disables top‑k)
    tk = -1 if top_k is None else top_k
    return GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=tk,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )

def generate_with_retry(
    model: nn.Module,
    enc: Dict[str, torch.Tensor],
    gen_cfg: GenerationConfig,
) -> torch.Tensor:
    """Call ``model.generate`` with fallback to greedy decoding on failure.

    Given an encoder output ``enc`` (e.g. produced by a tokenizer’s
    ``return_tensors="pt"`` call) and a pre‑constructed
    ``GenerationConfig``, this function attempts to generate a
    completion using sampling.  If a ``RuntimeError`` occurs (for
    example due to an invalid multinomial distribution), the function
    logs the error and retries with deterministic greedy decoding
    (``do_sample=False``).  This ensures that the training loop can
    continue even when sampling occasionally fails.

    Args:
        model: A ``PreTrainedModel`` instance supporting the
            ``generate`` method.
        enc: Dictionary containing at least ``input_ids`` and
            optionally ``attention_mask``.
        gen_cfg: A ``GenerationConfig`` controlling sampling.

    Returns:
        A tensor of generated token IDs.
    """
    try:
        return model.generate(**enc, generation_config=gen_cfg, synced_gpus=False)
    except RuntimeError as e:
        logging.error(f"generation failed with {e}; retrying greedily")
        return model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=gen_cfg.max_new_tokens,
            pad_token_id=gen_cfg.pad_token_id,
            eos_token_id=gen_cfg.eos_token_id,
            synced_gpus=False,
        )

# ----------------------------------------------------------------------------
# vLLM integration
# ----------------------------------------------------------------------------

try:
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    _HAS_VLLM = True
except Exception:
    _HAS_VLLM = False


class VLLMManager:
    """Asynchronous interface to a vLLM engine.

    When enabled, this class offloads text generation to a vLLM server
    running in the same process.  This can dramatically speed up
    generation, especially when using high parallelism, but it is
    currently incompatible with bitsandbytes quantisation.  The manager
    handles initialization, sleeping/waking, and weight loading.
    """

    def __init__(self, cfg: "DrGRPOConfig", rank: int):
        self.cfg = cfg
        self.rank = rank
        self.engine: Optional["AsyncLLMEngine"] = None

    async def init(self) -> None:
        """Initialize the underlying vLLM engine."""
        if self.engine is not None:
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
        if self.engine is None:
            raise RuntimeError("vLLM engine not initialised")
        sp = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_tokens,
        )
        res = await self.engine.generate(prompts, sp)
        outs: List[str] = []
        for r in res:
            t = r.outputs[0].text if r.outputs else ""
            t = _clip_think_block(t, self.cfg.max_think_tokens)
            outs.append(t)
        return outs

    async def sleep(self) -> None:
        if self.engine:
            await self.engine.sleep(level=2)

    async def wake(self) -> None:
        if self.engine:
            await self.engine.wake_up()

    def load_weights(self, sd: Dict[str, torch.Tensor]) -> None:
        if self.engine is None:
            raise RuntimeError("vLLM engine not initialised")
        core = self.engine.llm_engine.model_executor.driver_worker.model  # type: ignore
        core.load_weights(sd.items())

# ----------------------------------------------------------------------------
# Quantisation helpers
# ----------------------------------------------------------------------------

try:
    from transformers import BitsAndBytesConfig  # type: ignore
    import bitsandbytes as bnb
    from bitsandbytes.optim import PagedAdamW8bit  # type: ignore
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None  # type: ignore
    bnb = None
    PagedAdamW8bit = None  # type: ignore
    _HAS_BNB = False

# ----------------------------------------------------------------------------
# Liger kernel patching
# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
# Turn batcher (environment) import
# ----------------------------------------------------------------------------

try:
    # The environment used by ToSSim.  This is only available on rank 0.
    from Simulation.runner_with_rewards import RunnerWithRewards as TurnBatcher

except Exception:
    TurnBatcher = None  # type: ignore

# ----------------------------------------------------------------------------
# Configuration dataclasses
# ----------------------------------------------------------------------------

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
    sync_state: str = "SHARDED"  # SHARDED is cheaper for periodic vLLM syncs
    mixed_precision: bool = True
    activation_checkpointing: bool = True
    use_orig_params: bool = True
    offload_full_state_dict_to_cpu: bool = True
    rank0_only_state_dict: bool = True
    checkpoint_dir: Optional[str] = "checkpoints"


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
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    llm_int8_threshold: float = 6.0
    llm_int8_has_fp16_weight: bool = False

    def to_bnb(self):
        """Convert quantization arguments to a BitsAndBytesConfig."""
        if BitsAndBytesConfig is None:
            raise RuntimeError("You need a recent transformers for BitsAndBytesConfig")
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
    """Top‑level configuration for the GRPO trainer."""
    # Core model and training hyper‑parameters
    model_name: str = "google/gemma-3-27b-it"
    learning_rate: float = 5.0e-5
    min_learning_rate: float | None = 4.0e-5
    warmup_ticks: int = 100
    gradient_clip_norm: float = 1.0
    batch_size: int = 15
    max_iterations: int = 5000
    log_interval: int = 2
    k_per_prompt: int = 4
    beta: float = 0.0
    sync_frequency: int = 10
    use_flash_attention: bool = True
    use_liger: bool = False

    # Environment / game
    num_games: int = 2
    active_seats_per_game: int = 3

    # Sampling
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: Optional[int] = None
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 96
    max_think_tokens: Optional[int] = 64

    # Early stopping
    early_stop_parse_rate: Optional[float] = 0.99
    early_stop_consecutive_ticks: int = 1000

    # Loss variant and reward scaling
    loss_type: str = "dr_grpo"  # "grpo" or "dr_grpo"
    scale_rewards: bool = True

    # Switches
    disable_vllm: bool = False
    only_transformers: bool = True

    # Quantisation
    quant_mode: str = "4bit"  # bf16 | fp16 | 8bit | 4bit
    quant_args: QuantArgs = field(default_factory=QuantArgs)

    # Logging / outputs
    csv_path: str = "gemma3-27b_tos_training.csv"
    track_weight_deltas: bool = True
    sample_log_interval: int = 100
    sample_log_n: int = 2
    samples_path: str = "grpo_samples.jsonl"

    # Push to hub
    output_hub_repo: Optional[str] = None
    hub_private: bool = False
    max_shard_size: str = "10GB"

    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    vllm_config: VLLMConfig = field(default_factory=VLLMConfig)

    @classmethod
    def from_yaml(cls, p: str | Path) -> "DrGRPOConfig":
        """Load a configuration from a YAML file."""
        import yaml  # local import to avoid unnecessary dependency when not needed
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


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def _gemma_layer(model: nn.Module) -> type:
    """Return the decoder layer class for Gemma models."""
    for m in model.modules():
        if m.__class__.__name__.endswith("DecoderLayer"):
            return m.__class__
    raise RuntimeError("Cannot find Gemma DecoderLayer class")


def _clip_think_block(text: str, max_think_tokens: Optional[int]) -> str:
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


# ----------------------------------------------------------------------------
# CSV Logger
# ----------------------------------------------------------------------------

class CSVLogger:
    """Simple CSV logger that writes only on rank 0."""

    def __init__(self, path: str, rank: int) -> None:
        self.rank = rank
        if rank == 0:
            self.f = open(path, "w", newline="")
            self.w = csv.writer(self.f)
            self.w.writerow(["step", "loss", "avg_reward", "parse_acc", "weight_delta"])
            self.f.flush()
        else:
            self.f = None
            self.w = None

    def log(self, row: List[Any]) -> None:
        if self.rank == 0 and self.w is not None:
            self.w.writerow(row)
            self.f.flush()

    def close(self) -> None:
        if self.f:
            self.f.close()


# ----------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------

class Trainer:
    """Main GRPO training loop."""

    def __init__(self, cfg: DrGRPOConfig) -> None:
        self.cfg = cfg
        self._init_dist()

        # Auto‑disable vLLM if quantised or only_transformers
        if cfg.quant_mode.lower() in {"4bit", "8bit"}:
            if not cfg.only_transformers:
                logging.warning(
                    "quant_mode is %s → forcing disable_vllm=True (vLLM can't use bitsandbytes) and suggesting only_transformers=True",
                    cfg.quant_mode,
                )
            self.cfg.disable_vllm = True
        if cfg.only_transformers:
            logging.warning(
                "only_transformers=True → using pure HF + DDP (no FSDP, no vLLM)."
            )
            self.cfg.disable_vllm = True
        elif cfg.disable_vllm:
            logging.warning("disable_vllm=True → running FSDP with HF.generate (no vLLM).")

        # CSV logger and environment (only on rank 0)
        self.logger = CSVLogger(cfg.csv_path, self.rank)
        self.env: Optional[TurnBatcher] = None
        if self.rank == 0 and TurnBatcher is not None:
            self.env = TurnBatcher(cfg.num_games, cfg.active_seats_per_game, cfg.model_name)

        # Build model (FSDP or DDP) with quantization
        self._init_model()

        # vLLM manager (if enabled)
        self.vllm: Optional[VLLMManager] = None
        if not (self.cfg.disable_vllm or self.cfg.only_transformers or self.cfg.quant_mode.lower() in {"4bit", "8bit"}):
            self.vllm = VLLMManager(cfg, self.rank)

        self._maybe_track_init_weights()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._build_sched()
        self.step = 0

    # ---------------------------------------------------------------------
    # Distributed initialisation
    # ---------------------------------------------------------------------
    def _init_dist(self) -> None:
        """Initialise torch.distributed if necessary and set the device."""
        if not dist.is_initialized():
            dist.init_process_group("nccl", timeout=timedelta(hours=6))
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        torch.cuda.set_device(self.rank)

    # ---------------------------------------------------------------------
    # Model initialisation
    # ---------------------------------------------------------------------
    def _quant_config(self):
        """Return quantisation configuration or None for full precision."""
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

        # Build kwargs for model loading based on quantisation
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
        quantised = self.cfg.quant_mode.lower() in {"4bit", "8bit"}

        # DDP path for quantised or only_transformers
        if self.cfg.only_transformers or quantised:
            # Ensure model is on the correct device for FlashAttention‑2
            base.to(device)
            self.ddp = DDP(
                base,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False,
            )
            self.fsdp = None
            self.net: nn.Module = self.ddp  # forward path
            self.core: nn.Module = self.ddp.module  # raw model
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

    # ---------------------------------------------------------------------
    # Optimiser and scheduler
    # ---------------------------------------------------------------------
    def params(self):
        return self.net.parameters()

    def _init_optimizer(self):
        """Initialise the optimiser."""
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

        def decay(i: int) -> float:
            if i < warm:
                return i / max(1, warm)
            prog = (i - warm) / max(1, total - warm)
            return max(minlr / self.cfg.learning_rate, 0.5 * (1 + math.cos(math.pi * prog)))

        return LambdaLR(self.optimizer, decay)

    # ---------------------------------------------------------------------
    # Synchronisation helpers
    # ---------------------------------------------------------------------
    def _sync_bool(self, flag: bool) -> bool:
        """Synchronise a boolean flag across all ranks using all_reduce."""
        t = torch.tensor([1 if flag else 0], dtype=torch.int, device=f"cuda:{self.rank}")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.item() > 0

    # ---------------------------------------------------------------------
    # Weight deltas for monitoring
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Push to Hugging Face Hub
    # ---------------------------------------------------------------------
    def _push_to_hub(self) -> None:
        """Save and upload the model to the Hugging Face Hub."""
        if self.cfg.output_hub_repo is None:
            if self.rank == 0:
                logging.info("output_hub_repo not set — skipping push_to_hub.")
            return
        # Synchronise before exporting
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
                state_dict = self.core.state_dict()
                self.core.save_pretrained(
                    tmp,
                    state_dict=state_dict,
                    safe_serialization=True,
                    max_shard_size=self.cfg.max_shard_size,
                )
            else:
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
            from huggingface_hub import create_repo, upload_folder  # local import

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

    # ---------------------------------------------------------------------
    # Sample logging
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
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
                    logging.info(
                        f"Step {it}: loss={stats['loss']:.4f}, avg_reward={stats['avg_reward']:.4f}, parse_acc={stats['parse_acc']:.4f}"
                    )
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
                # Synchronise stop flag across ranks
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
                        cpu_model = self.core.to("cpu").float()
                        torch.save(cpu_model.state_dict(), ckdir / "fatal_ckpt_ddp.pt")
                    else:
                        fsd_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                        with FSDP.state_dict_type(
                            self.fsdp, StateDictType.FULL_STATE_DICT, fsd_cfg
                        ):
                            sd = {k: v.to("cpu") for k, v in self.fsdp.state_dict().items()}
                        torch.save(sd, ckdir / "fatal_ckpt_fsdp.pt")
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
                # Push model to hub if training finished normally
                self._push_to_hub()
                # Optionally plot metrics
                try:
                    import pandas as pd  # type: ignore
                    import matplotlib.pyplot as plt  # type: ignore

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

    # ---------------------------------------------------------------------
    # Single training step
    # ---------------------------------------------------------------------
    async def _step(self) -> Dict[str, Any]:
        """Perform a single training step and return logging stats."""
        # Get batch on rank 0 and broadcast to others
        if self.rank == 0:
            prompts, meta = self.env.next_batch() if self.env is not None else ([], [])
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
            rewards = self.env.apply_actions(meta * K, outs) if self.env else []
        else:
            rewards = []
        rewards = self._bcast(rewards)
        # Sample log
        if self.rank == 0 and self.step % self.cfg.sample_log_interval == 0:
            self._maybe_log_samples(tp, outs, rewards)
        # Loss
        lp = self._logprob_sums(tp, outs)
        loss = self._grpo_loss(
            lp,
            torch.tensor(rewards, device=lp.device, dtype=lp.dtype),
            group_ids,
            loss_type=("dr_grpo" if self.cfg.loss_type.lower() == "dr_grpo" else "grpo"),
            scale_rewards=self.cfg.scale_rewards,
        )
        # Optimiser step
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
        if self.rank == 0 and self.env is not None:
            parse_acc = self.env.get_batch_stats()["parsing_accuracy"]
        self.logger.log([self.step, loss_val, rew_avg, parse_acc, delta])
        return {
            "step": self.step,
            "loss": loss_val,
            "avg_reward": rew_avg,
            "parse_acc": parse_acc,
            "weight_delta": delta,
        }

    # ---------------------------------------------------------------------
    # HF-only generation path
    # ---------------------------------------------------------------------
    def _generate_with_hf(self, prompts: List[str]) -> List[str]:
        """Generate completions using the HuggingFace model directly."""
        # On non‑zero ranks, wait for rank 0 to finish
        if self.rank != 0:
            if dist.is_initialized():
                dist.barrier()
            return self._bcast_obj([], src=0)
        self.net.eval()
        # Build conservative generation config using parameters from cfg
        gen_cfg = build_generation_config(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            min_p=self.cfg.min_p,
            repetition_penalty=self.cfg.repetition_penalty,
            max_new_tokens=self.cfg.max_tokens,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        outs: List[str] = []
        device = next(self.params()).device
        with torch.no_grad():
            for p in prompts:
                enc = self.tok(p, return_tensors="pt").to(device)
                # Use generate_with_retry to handle NaN/inf or invalid distributions
                out_ids = generate_with_retry(self.core, enc, gen_cfg)
                # Decode only the new tokens
                text = self.tok.decode(
                    out_ids[0][enc["input_ids"].shape[1] :],
                    skip_special_tokens=False,
                )
                text = _clip_think_block(text, self.cfg.max_think_tokens)
                outs.append(text)
        self.net.train()
        if dist.is_initialized():
            dist.barrier()
        return self._bcast_obj(outs, src=0)

    # ---------------------------------------------------------------------
    # Weight synchronisation for vLLM
    # ---------------------------------------------------------------------
    async def _sync_weights(self) -> None:
        """Synchronise weights from FSDP to vLLM."""
        if self.vllm is None or self.fsdp is None:
            return
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

    # ---------------------------------------------------------------------
    # Broadcast helpers
    # ---------------------------------------------------------------------
    def _bcast(self, obj: Any, src: int = 0) -> Any:
        buf: List[Any] = [obj] if self.rank == src else [None]
        if dist.is_initialized():
            dist.broadcast_object_list(buf, src=src)
        return buf[0]

    def _bcast_obj(self, obj: Any, src: int = 0) -> Any:
        buf: List[Any] = [obj] if self.rank == src else [None]
        if dist.is_initialized():
            dist.broadcast_object_list(buf, src=src)
        return buf[0]

    def _mean(self, x: float) -> float:
        t = torch.tensor([x], device=f"cuda:{self.rank}")
        dist.all_reduce(t)
        return (t / self.world).item()

    # ---------------------------------------------------------------------
    # Log probability sums
    # ---------------------------------------------------------------------
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
        logits = self.net(**enc).logits.float()
        # Replace NaNs or Infs with a large negative value
        bad = ~torch.isfinite(logits)
        if bad.any():
            logits = torch.where(bad, torch.full_like(logits, -1e9), logits)
        res: List[torch.Tensor] = []
        for i, pl in enumerate(plens):
            start = max(pl - 1, 0)
            lp_slice = logits[i, start:-1]
            ids_slice = enc["input_ids"][i, pl:]
            if ids_slice.numel() == 0:
                res.append(torch.tensor(0.0, device=device, dtype=logits.dtype))
                continue
            log_probs = torch.log_softmax(lp_slice, dim=-1)
            log_probs = torch.nan_to_num(log_probs, nan=-1e9, neginf=-1e9, posinf=0.0)
            token_ll = log_probs.gather(1, ids_slice.unsqueeze(-1)).squeeze(-1)
            res.append(token_ll.sum())
        return torch.stack(res)

    # ---------------------------------------------------------------------
    # GRPO loss
    # ---------------------------------------------------------------------
    def _grpo_loss(
        self,
        lp: torch.Tensor,
        rw: torch.Tensor,
        gid: torch.Tensor,
        *,
        loss_type: str = "grpo",
        scale_rewards: bool = True,
    ) -> torch.Tensor:
        """Compute the GRPO/Dr.GRPO loss.

        Args:
            lp: log‑probability sums for each completion.
            rw: rewards tensor.
            gid: tensor assigning each completion to a prompt (group).
            loss_type: "grpo" for sequence‑length normalisation or
                "dr_grpo" to divide by a constant (max completion length).
            scale_rewards: if True, divide the advantages by the std. deviation.

        Returns:
            The scalar loss tensor.
        """
        device = lp.device
        adv = torch.zeros_like(rw, device=device, dtype=lp.dtype)
        # Compute group‑wise advantages
        for g in torch.unique(gid):
            m = gid == g
            rewards_g = rw[m]
            mean = rewards_g.mean()
            std = rewards_g.std() if scale_rewards else 1.0
            std = std if std > 0 else 1.0
            adv[m] = (rewards_g - mean) / std
        # Optionally normalise by maximum length (Dr. GRPO)
        if loss_type == "dr_grpo":
            max_len = max(1, self.cfg.max_tokens)
            return -(adv * lp / max_len).mean()
        return -(adv * lp).mean()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

async def _amain() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = DrGRPOConfig.from_yaml(args.config)
    tr = Trainer(cfg)
    await tr.train()


if __name__ == "__main__":
    asyncio.run(_amain())