# -*- coding: utf-8 -*-
"""
Dr GRPO (Group Relative Policy Optimization) – 2nd-gen rewrite (+ Push-to-Hub)
==============================================================================

Features:
- Hugging Face Hub export (rank-0 only)
- Selectable FULL / SHARDED state-dict sync from FSDP → vLLM
- Fault tolerance, CSV logging, PNG plotting
- Real-time weight-delta tracking
- Early stopping on parse rate
- Optional *pure FSDP* mode (no vLLM) via `disable_vllm`
- Optional *pure Transformers + DDP* mode via `only_transformers`
- Optional think-token clipping
- tqdm progress

This file MERGES everything so nothing is lost.
"""
from __future__ import annotations

import os
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_DEBUG", "INFO")  # or WARN if too chatty

import math, yaml, csv, json, logging, asyncio, functools, \
       tempfile, traceback, shutil, re
from pathlib import Path
from dataclasses import dataclass, field, fields as dc_fields
from typing import Any, Dict, List, Optional

from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision, ShardingStrategy, StateDictType,
    FullStateDictConfig, ShardedStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from huggingface_hub import create_repo, upload_folder

# ToSSim deps
from Simulation.turn_batcher import TurnBatcher  # type: ignore


# ============================ Configs ========================================

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
    sync_state: str = "SHARDED"           # FULL | SHARDED for vLLM sync
    mixed_precision: bool = True
    activation_checkpointing: bool = True
    use_orig_params: bool = True
    offload_full_state_dict_to_cpu: bool = True
    rank0_only_state_dict: bool = True
    checkpoint_dir: str | None = "checkpoints"


@dataclass
class VLLMConfig:
    tensor_parallel_size: int = 1
    max_num_seqs: int = 32
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.4
    enforce_eager: bool = True
    disable_log_stats: bool = True


@dataclass
class DrGRPOConfig:
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

    # visibility / samples
    sample_log_interval: int = 100     # every N optimizer steps
    sample_log_n: int = 2              # how many examples to print/save
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
    disable_vllm: bool = False           # run without vLLM, still FSDP
    only_transformers: bool = False      # run with pure HF + DDP (no FSDP, no vLLM)

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
        """Tolerant to extra keys in YAML."""
        raw = yaml.safe_load(Path(p).read_text())
        fsdp_raw = raw.pop("fsdp_config", {})
        vllm_raw = raw.pop("vllm_config", {})
        valid = {f.name for f in dc_fields(cls)}
        main_kwargs = {k: v for k, v in raw.items() if k in valid}
        fsdp = FSDPConfig(**fsdp_raw)
        vllm = VLLMConfig(**vllm_raw)
        return cls(**main_kwargs, fsdp_config=fsdp, vllm_config=vllm)


# ============================ Helpers ========================================

def _gemma_layer(model: nn.Module):
    for m in model.modules():
        if m.__class__.__name__.endswith("DecoderLayer"):
            return m.__class__
    raise RuntimeError("Cannot find Gemma DecoderLayer class")

def _clip_think_block(text: str, max_think_tokens: int | None) -> str:
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


# ============================ vLLM Manager ===================================

class VLLMManager:
    def __init__(self, cfg: DrGRPOConfig, rank: int):
        self.cfg = cfg
        self.rank = rank
        self.engine: AsyncLLMEngine | None = None

    async def init(self):
        if self.engine:
            return
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
        sp = SamplingParams(temperature=self.cfg.temperature,
                            top_p=self.cfg.top_p,
                            max_tokens=self.cfg.max_tokens)
        res = await self.engine.generate(prompts, sp)
        outs = []
        for r in res:
            t = r.outputs[0].text if r.outputs else ""
            t = _clip_think_block(t, self.cfg.max_think_tokens)
            outs.append(t)
        return outs

    async def sleep(self):
        if self.engine: await self.engine.sleep(level=2)

    async def wake(self):
        if self.engine: await self.engine.wake_up()

    def load_weights(self, sd: Dict[str, torch.Tensor]):
        core = (self.engine.llm_engine.model_executor.driver_worker.model)  # type: ignore
        core.load_weights(sd.items())


# ============================ CSV Logger =====================================

class CSVLogger:
    def __init__(self, path: str, rank: int):
        self.rank = rank
        if rank == 0:
            self.f = open(path, "w", newline="")
            self.w = csv.writer(self.f)
            self.w.writerow(["step", "loss", "avg_reward", "parse_acc", "weight_delta"])
            self.f.flush()
        else:
            self.f = None; self.w = None

    def log(self, row):
        if self.rank == 0:
            self.w.writerow(row); self.f.flush()

    def close(self):
        if self.f: self.f.close()


# ============================ Trainer ========================================

class Trainer:
    def __init__(self, cfg: DrGRPOConfig):
        self.cfg = cfg
        self._init_dist()

        if cfg.only_transformers:
            logging.warning("only_transformers=True → using pure HF + DDP (no FSDP, no vLLM).")
            self.cfg.disable_vllm = True  # force
        elif cfg.disable_vllm:
            logging.warning("disable_vllm=True → running FSDP with HF.generate (no vLLM).")

        self.logger = CSVLogger(cfg.csv_path, self.rank)
        self.env: Optional[TurnBatcher] = TurnBatcher(
            cfg.num_games, cfg.active_seats_per_game, cfg.model_name
        ) if self.rank == 0 else None

        # Build model (FSDP or DDP)
        self._init_model()

        # vLLM only if allowed
        self.vllm = None if (cfg.disable_vllm or cfg.only_transformers) else VLLMManager(cfg, self.rank)

        self._maybe_track_init_weights()
        self.optimizer = torch.optim.AdamW(self.params(), lr=cfg.learning_rate)
        self.scheduler = self._build_sched()
        self.step = 0

    # ---- dist ----
    def _init_dist(self):
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        torch.cuda.set_device(self.rank)

    # ---- model ----
    def _init_model(self):
        self.tok = AutoTokenizer.from_pretrained(self.cfg.model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name, torch_dtype=_dtype(), low_cpu_mem_usage=True
        )

        device = torch.device(f"cuda:{self.rank}")

        if self.cfg.only_transformers:
            # ---- Pure HF + DDP path ----
            base.to(device)
            self.ddp = DDP(base, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=False)
            self.fsdp = None
            self.net = self.ddp                 # forward path
            self.core = self.ddp.module         # raw model for .generate() / save
        else:
            # ---- FSDP path ----
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
                mixed_precision=_build_mp(_dtype()) if self.cfg.fsdp_config.mixed_precision else None,
                device_id=torch.cuda.current_device(),
                use_orig_params=self.cfg.fsdp_config.use_orig_params,
            )
            self.ddp = None
            self.net = self.fsdp                # forward path
            self.core = self.fsdp.module        # raw model for .generate() / save

    def params(self):
        return self.net.parameters()

    def _build_sched(self):
        warm, total, minlr = self.cfg.warmup_ticks, self.cfg.max_iterations, (self.cfg.min_learning_rate or 0.0)
        def decay(i):
            if i < warm: return i / max(1, warm)
            prog = (i - warm) / max(1, total - warm)
            return max(minlr / self.cfg.learning_rate, 0.5 * (1 + math.cos(math.pi * prog)))
        return LambdaLR(self.optimizer, decay)

    # ---- weight deltas ----
    def _maybe_track_init_weights(self):
        self.track = self.cfg.track_weight_deltas and (self.rank == 0)
        if self.track:
            self.init_vec = torch.nn.utils.parameters_to_vector(
                [p.detach().cpu() for p in self.params()]
            )
        else:
            self.init_vec = None

    def _current_delta(self) -> float:
        if not self.track: return 0.0
        cur = torch.nn.utils.parameters_to_vector(
            [p.detach().cpu() for p in self.params()]
        )
        return torch.norm(cur - self.init_vec).item()

    # ---- hub push ----
    def _push_to_hub(self):
        if self.cfg.output_hub_repo is None:
            if self.rank == 0:
                logging.info("output_hub_repo not set — skipping push_to_hub.")
            return

        dist.barrier()
        if self.rank != 0:
            dist.barrier()
            return

        logging.info("Exporting state_dict for push_to_hub …")

        tmp = tempfile.mkdtemp()
        try:
            if self.cfg.only_transformers:
                # plain DDP state_dict
                state_dict = self.core.state_dict()
                self.core.save_pretrained(
                    tmp,
                    state_dict=state_dict,
                    safe_serialization=True,
                    max_shard_size=self.cfg.max_shard_size,
                )
            else:
                # FSDP FULL state
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
            create_repo(self.cfg.output_hub_repo, exist_ok=True,
                        repo_type="model", private=self.cfg.hub_private)
            upload_folder(repo_id=self.cfg.output_hub_repo,
                          folder_path=tmp, repo_type="model")
            logging.info("Push to Hub complete.")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        dist.barrier()

    # ---- sample logging ----
    def _maybe_log_samples(self, prompts, outs, rewards):
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
                    i, self.step, prompts[i], outs[i], rec["reward"]
                )

    # ==================== training ====================
    async def train(self):
        if self.vllm is not None:
            await self.vllm.init()

        consecutive_hits = 0
        total_steps = self.cfg.max_iterations

        try:
            for it in trange(total_steps, desc="Training", disable=(self.rank != 0)):
                self.step = it
                stats = await self._step()

                if self.rank == 0:
                    tqdm.write(f"Step {it}: loss={stats['loss']:.4f}, avg_reward={stats['avg_reward']:.4f}, parse_acc={stats['parse_acc']:.4f}")
                    if it % self.cfg.log_interval == 0:
                        logging.info(json.dumps(stats))

                stop = False
                if (self.rank == 0 and
                    self.cfg.early_stop_parse_rate is not None and
                    self.cfg.early_stop_consecutive_ticks > 0):
                    current_acc = stats.get("parse_acc", 0.0)
                    consecutive_hits = consecutive_hits + 1 if current_acc >= self.cfg.early_stop_parse_rate else 0
                    if consecutive_hits >= self.cfg.early_stop_consecutive_ticks:
                        logging.info(
                            f"[early-stop] parse_acc {current_acc:.4f} for "
                            f"{consecutive_hits} consecutive steps. Stopping."
                        )
                        stop = True

                stop = self._bcast(stop)
                if stop:
                    break

        except Exception:
            if self.rank == 0:
                logging.error("Fatal error, saving checkpoint…")
                ckdir = Path(self.cfg.fsdp_config.checkpoint_dir or "checkpoints")
                ckdir.mkdir(exist_ok=True)
                try:
                    if self.cfg.only_transformers:
                        torch.save(self.core.state_dict(), ckdir / "fatal_ckpt_ddp.pt")
                    else:
                        fsd_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                        with FSDP.state_dict_type(self.fsdp, StateDictType.FULL_STATE_DICT, fsd_cfg):
                            torch.save(self.fsdp.state_dict(), ckdir / "fatal_ckpt_fsdp.pt")
                except Exception as e:
                    logging.error(f"Failed to save fatal checkpoint: {e}")
                logging.error(traceback.format_exc())
            raise
        finally:
            if self.rank == 0:
                self._push_to_hub()
                try:
                    import pandas as pd, matplotlib.pyplot as plt
                    df = pd.read_csv(self.cfg.csv_path)
                    plt.plot(df["step"], df["avg_reward"], label="avg_reward")
                    plt.plot(df["step"], df["parse_acc"], label="parse_acc")
                    plt.legend(); plt.xlabel("step")
                    plt.savefig("training_metrics.png"); plt.close()
                except Exception as e:
                    logging.warning(f"Plotting failed: {e}")
            self.logger.close()
            if dist.is_initialized():
                try:
                    dist.barrier()
                except Exception:
                    pass
                dist.destroy_process_group()

    async def _step(self):
        # get batch
        if self.rank == 0:
            prompts, meta = self.env.next_batch()
        else:
            prompts, meta = [], []
        prompts = self._bcast(prompts); meta = self._bcast(meta)
        if not prompts:
            return {
                "step": self.step, "loss": 0.0, "avg_reward": 0.0,
                "parse_acc": 0.0, "weight_delta": 0.0
            }

        # k sampling
        K = self.cfg.k_per_prompt
        tp = [p for p in prompts for _ in range(K)]
        group_ids = torch.arange(
            len(prompts), device=f"cuda:{self.rank}"
        ).repeat_interleave(K)

        # generation
        if self.cfg.disable_vllm or self.cfg.only_transformers:
            outs = self._generate_with_hf(tp)
        else:
            outs = await self.vllm.generate(tp)

        # rewards
        if self.rank == 0:
            rewards = [r for r, _ in self.env.apply_actions(meta * K, outs)]
        else:
            rewards = []
        rewards = self._bcast(rewards)

        # sample log
        if self.rank == 0 and self.step % self.cfg.sample_log_interval == 0:
            self._maybe_log_samples(tp, outs, rewards)

        # loss
        lp = self._logprob_sums(tp, outs)
        loss = self._grpo_loss(lp, torch.tensor(rewards, device=lp.device), group_ids)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params(), self.cfg.gradient_clip_norm)
        self.optimizer.step()
        self.scheduler.step()

        # sync weights → only if vLLM is alive
        if (self.vllm is not None) and (self.step % self.cfg.sync_frequency == 0):
            await self._sync_weights()

        # log
        loss_val = self._mean(loss.item())
        rew_avg = sum(rewards) / max(1, len(rewards))
        delta = self._current_delta()
        parse_acc = 0.0
        if self.rank == 0:
            parse_acc = self.env.get_batch_stats()["parsing_accuracy"]
        self.logger.log([self.step, loss_val, rew_avg, parse_acc, delta])
        return {"step": self.step, "loss": loss_val, "avg_reward": rew_avg,
                "parse_acc": parse_acc, "weight_delta": delta}

    # --- HF-only generation path (used for disable_vllm or only_transformers) ---
    def _generate_with_hf(self, prompts: List[str]) -> List[str]:
        # Rank-0 runs generation to avoid NCCL timeouts; broadcast to others.
        if self.rank != 0:
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
                enc = self.tok(p, return_tensors="pt").to(device)
                out_ids = self.core.generate(**enc, generation_config=gen_cfg)
                text = self.tok.decode(
                    out_ids[0][enc["input_ids"].shape[1]:],
                    skip_special_tokens=False
                )
                text = _clip_think_block(text, self.cfg.max_think_tokens)
                outs.append(text)
        self.net.train()
        return self._bcast_obj(outs, src=0)

    async def _sync_weights(self):
        # Only called if vLLM is enabled and FSDP mode is active
        mode = self.cfg.fsdp_config.sync_state.upper()
        await self.vllm.sleep(); dist.barrier()
        if mode == "FULL":
            cfg = FullStateDictConfig(offload_to_cpu=True,
                                      rank0_only=self.cfg.fsdp_config.rank0_only_state_dict)
            with FSDP.state_dict_type(self.fsdp, StateDictType.FULL_STATE_DICT, cfg):
                sd = self.fsdp.state_dict()
            target = sd
        else:  # SHARDED
            cfg = ShardedStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(self.fsdp, StateDictType.SHARDED_STATE_DICT, cfg):
                sd = self.fsdp.state_dict()
            target = {k: v for k, v in sd.items()}
        self.vllm.load_weights(target)
        await self.vllm.wake(); dist.barrier()

    # ---- utils ----
    def _bcast(self, obj, src=0):
        l = [obj] if self.rank == src else [None]
        dist.broadcast_object_list(l, src=src)
        return l[0]

    def _bcast_obj(self, obj, src=0):
        buf = [obj] if self.rank == src else [None]
        dist.broadcast_object_list(buf, src=src)
        return buf[0]

    def _mean(self, x: float):
        t = torch.tensor([x], device=f"cuda:{self.rank}")
        dist.all_reduce(t)
        return (t / self.world).item()

    def _logprob_sums(self, prompts, completions):
        device = next(self.params()).device
        maxlen = self.cfg.vllm_config.max_model_len
        full = [p + c for p, c in zip(prompts, completions)]
        plens = [len(self.tok.encode(p)) for p in prompts]
        enc = self.tok(full, padding=True, truncation=True, max_length=maxlen,
                       return_tensors="pt").to(device)
        with torch.cuda.amp.autocast(enabled=(not self.cfg.only_transformers)):
            logits = self.net(**enc).logits
        res = []
        for i, pl in enumerate(plens):
            lp = torch.log_softmax(logits[i, pl - 1:-1], -1)
            idx = enc["input_ids"][i, pl:]
            res.append(lp.gather(1, idx.unsqueeze(-1)).squeeze(-1).sum())
        return torch.stack(res)

    def _grpo_loss(self, lp, rw, gid):
        adv = torch.zeros_like(rw)
        for g in torch.unique(gid):
            m = (gid == g)
            adv[m] = rw[m] - rw[m].mean()
        return -(adv * lp).mean()


# ============================ CLI ============================================

async def _amain():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--log-level', default='INFO')
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format='%(asctime)s %(levelname)s %(message)s')
    cfg = DrGRPOConfig.from_yaml(args.config)
    tr = Trainer(cfg)
    await tr.train()

if __name__ == '__main__':
    asyncio.run(_amain())
