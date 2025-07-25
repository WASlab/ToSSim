
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
from typing import Any, Dict, List, Optional, Union

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

# Environment knobs – keep them the same but harmless when dist is off
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_DEBUG", "INFO")

# ---------------------------------------------------------------------------
# Safe multinomial & generation helpers
# ---------------------------------------------------------------------------
from Simulation.sequential_env import SequentialGRPOEnv

_orig_multinomial = torch.multinomial

def safe_multinomial(
    input: torch.Tensor,
    num_samples: int,
    replacement: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not torch.is_floating_point(input):
        input = input.float()
    clean = torch.nan_to_num(input, nan=0.0, posinf=0.0, neginf=0.0)
    clean = torch.clamp(clean, min=0.0)
    if clean.dim() == 1:
        if clean.sum() <= 0:
            clean = torch.ones_like(clean)
    else:
        sums = clean.sum(dim=-1, keepdim=True)
        mask = sums <= 0
        if mask.any():
            uniform = torch.ones_like(clean)
            clean = torch.where(mask, uniform, clean)
    return _orig_multinomial(clean, num_samples, replacement=replacement, generator=generator)

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

def generate_with_retry(model: nn.Module, enc: Dict[str, torch.Tensor], gen_cfg: GenerationConfig) -> torch.Tensor:
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

# ---------------------------------------------------------------------------
# vLLM (optional)
# ---------------------------------------------------------------------------
try:
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    _HAS_VLLM = True
except Exception:
    _HAS_VLLM = False

class VLLMManager:
    def __init__(self, cfg: "DrGRPOConfig", rank: int):
        self.cfg = cfg
        self.rank = rank
        self.engine: Optional["AsyncLLMEngine"] = None

    async def init(self) -> None:
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

# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Liger kernel
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def _dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def _build_mp(dtype: torch.dtype) -> MixedPrecision:
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

def _sharding(name: str) -> ShardingStrategy:
    name = name.upper()
    return {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }[name]

@dataclass
class FSDPConfig:
    sharding_strategy: str = "FULL_SHARD"
    sync_state: str = "SHARDED"
    mixed_precision: bool = True
    activation_checkpointing: bool = True
    use_orig_params: bool = True
    offload_full_state_dict_to_cpu: bool = True
    rank0_only_state_dict: bool = True
    checkpoint_dir: Optional[str] = "checkpoints"

@dataclass
class VLLMConfig:
    tensor_parallel_size: int = 1
    max_num_seqs: int = 32
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.4
    enforce_eager: bool = True
    disable_log_stats: bool = True

@dataclass
class QuantArgs:
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    llm_int8_threshold: float = 6.0
    llm_int8_has_fp16_weight: bool = False
    def to_bnb(self):
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
    # Core
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

    # Env
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

    # Early stop
    early_stop_parse_rate: Optional[float] = 0.99
    early_stop_consecutive_ticks: int = 1000

    # Loss
    loss_type: str = "dr_grpo"
    scale_rewards: bool = True

    # Switches
    disable_vllm: bool = False
    only_transformers: bool = True

    # **NEW** — single-GPU force
    force_single_gpu: bool = False
    single_device: int = 0

    # Quant
    quant_mode: str = "4bit"  # bf16 | fp16 | 8bit | 4bit
    quant_args: QuantArgs = field(default_factory=QuantArgs)

    # Logging
    csv_path: str = "gemma3-27b_tos_training.csv"
    track_weight_deltas: bool = True
    sample_log_interval: int = 100
    sample_log_n: int = 2
    samples_path: str = "grpo_samples.jsonl"

    # Hub
    output_hub_repo: Optional[str] = None
    hub_private: bool = False
    max_shard_size: str = "10GB"

    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    vllm_config: VLLMConfig = field(default_factory=VLLMConfig)

    @classmethod
    def from_yaml(cls, p: str | Path) -> "DrGRPOConfig":
        import yaml
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gemma_layer(model: nn.Module) -> type:
    for m in model.modules():
        if m.__class__.__name__.endswith("DecoderLayer"):
            return m.__class__
    raise RuntimeError("Cannot find Gemma DecoderLayer class")

def _clip_think_block(text: str, max_think_tokens: Optional[int]) -> str:
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

# ---------------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------------
class CSVLogger:
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

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg: DrGRPOConfig, env: Optional[Any] = None) -> None:
        self.cfg = cfg
        self._init_dist()

        # Safety switches for vLLM & quant
        if cfg.quant_mode.lower() in {"4bit", "8bit"}:
            if not cfg.only_transformers:
                logging.warning("quant_mode=%s → forcing disable_vllm=True", cfg.quant_mode)
            self.cfg.disable_vllm = True
        if cfg.only_transformers:
            logging.warning("only_transformers=True → using pure HF (no FSDP, no vLLM).")
            self.cfg.disable_vllm = True
        elif cfg.disable_vllm:
            logging.warning("disable_vllm=True → running FSDP with HF.generate (no vLLM).")

        self.logger = CSVLogger(cfg.csv_path, self.rank)

        if env is None:
            self._init_env()
        else:
            self.env = env

        self._init_model()

        self.vllm: Optional[VLLMManager] = None
        if not (
            self.cfg.disable_vllm
            or self.cfg.only_transformers
            or self.cfg.quant_mode.lower() in {"4bit", "8bit"}
            or self.cfg.force_single_gpu
        ):
            self.vllm = VLLMManager(cfg, self.rank)

        self._maybe_track_init_weights()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._build_sched()
        self.step = 0

    # ---------------------------
    # Env init (optional)
    # ---------------------------
    def _init_env(self) -> None:
        self.env = None
        if self.rank == 0:
            try:
                env = SequentialGRPOEnv(
                    num_games=self.cfg.num_games,
                    active_seats_per_game=self.cfg.active_seats_per_game,
                    model_name=self.cfg.model_name,
                    evals_per_phase=3,
                    max_days=7,
                    prompts_per_call=1,
                )
                # Probe
                probes, _ = env.next_batch()
                if not probes:
                    logging.warning("Sanity probe: env.next_batch() returned 0 prompts.")
                self.env = env
            except Exception as e:
                logging.exception("Failed to construct environment: %s", e)
                self.env = None
        # broadcast presence (no-op in single-gpu mode)
        if dist.is_initialized():
            has_env = torch.tensor([1 if self.env is not None else 0], device=f"cuda:{self.rank}", dtype=torch.int)
            dist.broadcast(has_env, src=0)
            if has_env.item() == 0:
                self.env = None

    # ---------------------------
    # Dist init
    # ---------------------------
    def _init_dist(self) -> None:
        if getattr(self.cfg, "force_single_gpu", False):
            # Pure single-GPU path
            self.rank = 0
            self.world = 1
            torch.cuda.set_device(self.cfg.single_device)
            self.device = torch.device(f"cuda:{self.cfg.single_device}")
            logging.info("Running in force_single_gpu mode on cuda:%d", self.cfg.single_device)
            return
        if not dist.is_initialized():
            dist.init_process_group("nccl", timeout=timedelta(hours=6))
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        torch.cuda.set_device(self.rank)
        self.device = torch.device(f"cuda:{self.rank}")

    # ---------------------------
    # Model init
    # ---------------------------
    def _quant_config(self):
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
        self.tok = AutoTokenizer.from_pretrained(self.cfg.model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        quant = self._quant_config()
        load_kwargs: Dict[str, Any] = {}
        if quant is None:
            load_kwargs.update(dict(torch_dtype=_dtype(), low_cpu_mem_usage=True))
        elif quant == "8bit":
            load_kwargs.update(dict(load_in_8bit=True, device_map={"": self.cfg.single_device if self.cfg.force_single_gpu else self.rank}))
        else:
            load_kwargs.update(dict(quantization_config=quant, device_map={"": self.cfg.single_device if self.cfg.force_single_gpu else self.rank}))

        if self.cfg.use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"

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
            logging.warning("use_liger=True but liger_kernel is not installed; skipping.")

        # --------- SINGLE GPU EARLY OUT ---------
        if self.cfg.force_single_gpu:
            base = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **load_kwargs)
            base.to(self.device)
            self.net = base
            self.core = base
            self.ddp = None
            self.fsdp = None
            return
        # ----------------------------------------

        # Multi-GPU paths
        base = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **load_kwargs)
        device = self.device
        quantised = self.cfg.quant_mode.lower() in {"4bit", "8bit"}

        if self.cfg.only_transformers or quantised:
            base.to(device)
            self.ddp = DDP(
                base,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False,
            )
            self.fsdp = None
            self.net = self.ddp
            self.core = self.ddp.module
        else:
            if self.cfg.fsdp_config.activation_checkpointing:
                apply_activation_checkpointing(
                    base,
                    checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
                    check_fn=lambda m: m.__class__.__name__.endswith("DecoderLayer"),
                )
            awp = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={_gemma_layer(base)})
            self.fsdp = FSDP(
                base,
                auto_wrap_policy=awp,
                sharding_strategy=_sharding(self.cfg.fsdp_config.sharding_strategy),
                mixed_precision=_build_mp(_dtype()) if self.cfg.fsdp_config.mixed_precision else None,
                device_id=torch.cuda.current_device(),
                use_orig_params=self.cfg.fsdp_config.use_orig_params,
            )
            self.ddp = None
            self.net = self.fsdp
            self.core = self.fsdp.module

    # ---------------------------
    # Optim / sched
    # ---------------------------
    def params(self):
        return self.net.parameters()

    def _init_optimizer(self):
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

    # ---------------------------
    # Sync helpers
    # ---------------------------
    def _sync_bool(self, flag: bool) -> bool:
        if not dist.is_initialized():
            return flag
        t = torch.tensor([1 if flag else 0], dtype=torch.int, device=f"cuda:{self.rank}")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.item() > 0

    # ---------------------------
    # Weight deltas
    # ---------------------------
    def _maybe_track_init_weights(self) -> None:
        self.track = self.cfg.track_weight_deltas and (self.rank == 0)
        if self.track:
            self.init_vec = torch.nn.utils.parameters_to_vector([p.detach().cpu() for p in self.params()])
        else:
            self.init_vec = None
    def _current_delta(self) -> float:
        if not self.track:
            return 0.0
        cur = torch.nn.utils.parameters_to_vector([p.detach().cpu() for p in self.params()])
        return torch.norm(cur - self.init_vec).item()

    # ---------------------------
    # Push to Hub
    # ---------------------------
    def _push_to_hub(self) -> None:
        if self.cfg.output_hub_repo is None:
            if self.rank == 0:
                logging.info("output_hub_repo not set — skipping push_to_hub.")
            return
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
                with FSDP.state_dict_type(self.fsdp, StateDictType.FULL_STATE_DICT, fsd_cfg):
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
            upload_folder(repo_id=self.cfg.output_hub_repo, folder_path=tmp, repo_type="model")
            logging.info("Push to Hub complete.")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if dist.is_initialized():
            dist.barrier()

    # ---------------------------
    # Sample logging
    # ---------------------------
    def _maybe_log_samples(self, prompts: List[str], outs: List[str], rewards: List[float]) -> None:
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
                    i, self.step, prompts[i], outs[i], rec["reward"],
                )

    # ---------------------------
    # Train loop
    # ---------------------------
    async def train(self) -> None:
        if self.vllm is not None:
            await self.vllm.init()
        consecutive_hits = 0
        total_steps = self.cfg.max_iterations
        try:
            for it in trange(total_steps, desc="Training", disable=(self.rank != 0)):
                self.step = it
                stats = await self._step()
                if self.rank == 0:
                    logging.info(
                        f"Step {it}: loss={stats['loss']:.4f}, avg_reward={stats['avg_reward']:.4f}, parse_acc={stats['parse_acc']:.4f}"
                    )
                stop = False
                if (
                    self.rank == 0
                    and self.cfg.early_stop_parse_rate is not None
                    and self.cfg.early_stop_consecutive_ticks > 0
                ):
                    current_acc = stats.get("parse_acc", 0.0)
                    consecutive_hits = consecutive_hits + 1 if current_acc >= self.cfg.early_stop_parse_rate else 0
                    if consecutive_hits >= self.cfg.early_stop_consecutive_ticks:
                        logging.info(
                            f"[early-stop] parse_acc {current_acc:.4f} for {consecutive_hits} consecutive steps. Stopping."
                        )
                        stop = True
                stop = self._sync_bool(stop)
                if stop:
                    break
        except Exception:
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
                        with FSDP.state_dict_type(self.fsdp, StateDictType.FULL_STATE_DICT, fsd_cfg):
                            sd = {k: v.to("cpu") for k, v in self.fsdp.state_dict().items()}
                        torch.save(sd, ckdir / "fatal_ckpt_fsdp.pt")
                except Exception as e:
                    logging.error(f"Failed to save fatal checkpoint: {e}")
                logging.error(traceback.format_exc())
            with contextlib.suppress(Exception):
                if dist.is_initialized():
                    dist.destroy_process_group()
            raise
        finally:
            if self.rank == 0:
                self._push_to_hub()
                try:
                    import pandas as pd
                    import matplotlib.pyplot as plt
                    df = pd.read_csv(self.cfg.csv_path)
                    plt.plot(df["step"], df["avg_reward"], label="avg_reward")
                    plt.plot(df["step"], df["parse_acc"], label="parse_acc")
                    plt.legend(); plt.xlabel("step")
                    plt.savefig("training_metrics.png"); plt.close()
                except Exception as e:
                    logging.warning(f"Plotting failed: {e}")
            self.logger.close()
            if dist.is_initialized():
                with contextlib.suppress(Exception):
                    dist.barrier()
                with contextlib.suppress(Exception):
                    dist.destroy_process_group()

    # ---------------------------
    # Step
    # ---------------------------
    async def _step(self) -> Dict[str, Any]:
        if self.env is None:
            if self.rank == 0:
                logging.error("No environment attached. Training cannot proceed.")
            return {"step": self.step, "loss": 0.0, "avg_reward": 0.0, "parse_acc": 0.0, "weight_delta": 0.0}

        max_empty_loops = max(1, self.cfg.num_games)
        empty_loops = 0
        while True:
            if self.rank == 0:
                prompts, meta = self.env.next_batch()
            else:
                prompts, meta = [], []
            prompts = self._bcast(prompts)
            meta = self._bcast(meta)
            if prompts:
                break
            empty_loops += 1
            if empty_loops >= max_empty_loops:
                if self.rank == 0:
                    stats = self.env.get_batch_stats()
                    logging.warning(f"[step {self.step}] EMPTY step ({empty_loops} tries). env stats: {stats}")
                return {"step": self.step, "loss": 0.0, "avg_reward": 0.0, "parse_acc": 0.0, "weight_delta": 0.0}
            if dist.is_initialized():
                dist.barrier()

        K = self.cfg.k_per_prompt
        tp = [p for p in prompts for _ in range(K)]
        group_ids = torch.arange(len(prompts), device=self.device).repeat_interleave(K)

        if self.vllm is None:
            outs = self._generate_with_hf(tp)
        else:
            outs = await self.vllm.generate(tp)

        if self.rank == 0:
            rewards = self.env.apply_actions(meta * K, outs)
        else:
            rewards = []
        rewards = self._bcast(rewards)

        if self.rank == 0 and self.step % self.cfg.sample_log_interval == 0:
            self._maybe_log_samples(tp, outs, rewards)

        lp = self._logprob_sums(tp, outs)
        loss = self._grpo_loss(
            lp,
            torch.tensor(rewards, device=lp.device, dtype=lp.dtype),
            group_ids,
            loss_type=("dr_grpo" if self.cfg.loss_type.lower() == "dr_grpo" else "grpo"),
            scale_rewards=self.cfg.scale_rewards,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params(), self.cfg.gradient_clip_norm)
        self.optimizer.step()
        self.scheduler.step()

        if self.vllm is not None and (self.step % self.cfg.sync_frequency == 0):
            await self._sync_weights()

        loss_val = self._mean(loss.item()) if dist.is_initialized() else loss.item()
        rew_avg = sum(rewards) / max(1, len(rewards))
        delta = self._current_delta()
        parse_acc = 0.0
        if self.rank == 0:
            parse_acc = self.env.get_batch_stats()["parsing_accuracy"]

        self.logger.log([self.step, loss_val, rew_avg, parse_acc, delta])
        return {"step": self.step, "loss": loss_val, "avg_reward": rew_avg, "parse_acc": parse_acc, "weight_delta": delta}

    # ---------------------------
    # HF generation
    # ---------------------------
    def _generate_with_hf(self, prompts: List[str]) -> List[str]:
        if self.rank != 0 and dist.is_initialized():
            dist.barrier()
            return self._bcast_obj([], src=0)
        self.net.eval()
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
                out_ids = generate_with_retry(self.core, enc, gen_cfg)
                text = self.tok.decode(out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
                text = _clip_think_block(text, self.cfg.max_think_tokens)
                outs.append(text)
        self.net.train()
        if dist.is_initialized():
            dist.barrier()
        return self._bcast_obj(outs, src=0) if dist.is_initialized() else outs

    # ---------------------------
    # vLLM sync (multi-GPU only)
    # ---------------------------
    async def _sync_weights(self) -> None:
        if self.vllm is None or self.fsdp is None:
            return
        mode = self.cfg.fsdp_config.sync_state.upper()
        await self.vllm.sleep()
        if dist.is_initialized():
            dist.barrier()
        if mode == "FULL":
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=self.cfg.fsdp_config.rank0_only_state_dict)
            with FSDP.state_dict_type(self.fsdp, StateDictType.FULL_STATE_DICT, cfg):
                sd = self.fsdp.state_dict()
            target = sd
        else:
            cfg = ShardedStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(self.fsdp, StateDictType.SHARDED_STATE_DICT, cfg):
                sd = self.fsdp.state_dict()
            target = {k: v for k, v in sd.items()}
        self.vllm.load_weights(target)
        await self.vllm.wake()
        if dist.is_initialized():
            dist.barrier()

    # ---------------------------
    # Broadcast helpers
    # ---------------------------
    def _bcast(self, obj: Any, src: int = 0) -> Any:
        if not dist.is_initialized():
            return obj
        buf: List[Any] = [obj] if self.rank == src else [None]
        dist.broadcast_object_list(buf, src=src)
        return buf[0]

    def _bcast_obj(self, obj: Any, src: int = 0) -> Any:
        if not dist.is_initialized():
            return obj
        buf: List[Any] = [obj] if self.rank == src else [None]
        dist.broadcast_object_list(buf, src=src)
        return buf[0]

    def _mean(self, x: float) -> float:
        if not dist.is_initialized():
            return x
        t = torch.tensor([x], device=f"cuda:{self.rank}")
        dist.all_reduce(t)
        return (t / self.world).item()

    # ---------------------------
    # Log prob sums
    # ---------------------------
    def _logprob_sums(self, prompts: List[str], completions: List[str]) -> torch.Tensor:
        device = next(self.params()).device
        maxlen = min(
            self.cfg.vllm_config.max_model_len,
            getattr(self.core.config, "max_position_embeddings", 32768),
            self.tok.model_max_length,
        )
        texts = [p + c for p, c in zip(prompts, completions)]
        plens = [len(self.tok.encode(p)) for p in prompts]
        enc = self.tok(texts, padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to(device)
        logits = self.net(**enc).logits.float()
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

    # ---------------------------
    # GRPO loss
    # ---------------------------
    def _grpo_loss(
        self,
        lp: torch.Tensor,
        rw: torch.Tensor,
        gid: torch.Tensor,
        *,
        loss_type: str = "grpo",
        scale_rewards: bool = True,
    ) -> torch.Tensor:
        device = lp.device
        adv = torch.zeros_like(rw, device=device, dtype=lp.dtype)
        for g in torch.unique(gid):
            m = gid == g
            rewards_g = rw[m]
            mean = rewards_g.mean()
            std = rewards_g.std() if scale_rewards else 1.0
            std = std if std > 0 else 1.0
            adv[m] = (rewards_g - mean) / std
        if loss_type == "dr_grpo":
            max_len = max(1, self.cfg.max_tokens)
            return -(adv * lp / max_len).mean()
        return -(adv * lp).mean()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
async def _amain() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    cfg = DrGRPOConfig.from_yaml(args.config)
    tr = Trainer(cfg, env=None)
    await tr.train()

if __name__ == "__main__":
    asyncio.run(_amain())