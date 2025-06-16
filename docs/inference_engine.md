1:9999
# Inference Engine — Unified Specification (v0.3)

_Last updated : 2025-06-16_

The **Inference Engine** links the ToSSim game-loop to GPU-resident LLMs served
by [vLLM](https://github.com/vllm-project/vllm).  Its responsibilities:

1. Launch vLLM servers on every visible GPU (one _or_ two per GPU, §2).
2. Allocate each agent to a _lane_ so at most **two** checkpoints live on one
   GPU or MIG slice.
3. Serve prompts asynchronously, apply realistic latency, and return
   completions to the game-loop.
4. Emit timing events so replays can reproduce wall-clock order.

---

## 1 GPU layout options

| ID | Layout on a single H100 | Prefill HoL¹ | Latency isolation | Extra proc | Recommended use |
|----|-------------------------|--------------|-------------------|------------|-----------------|
| A | **1 vLLM** instance, 2 checkpoints | **Yes** – prefill blocks both models | weak | 0 | high-throughput offline sims |
| B | **2 vLLM** instances, same GPU | **No** — CUDA interleaves kernels | good | +1 | real-time chat (default) |
| C | Two **3g.40 GB MIG** slices, 1 server each | No | excellent (hard split) | +1 | benchmarking / fairness |

¹ Head-of-Line.  Layout B avoids it without extra VRAM; therefore the engine
spawns **two servers per GPU by default** and only enables MIG when
`--enable-mig` is passed at launch.

---

## 2 Resource limits (phase-1 deployment)

* ≤ 8 GPUs per node (H100 class)
* ≤ 2 resident agents / checkpoints per GPU **or** MIG slice
* If `logical_agents > 2 × GPUs` the engine **swaps** weights (≈1-2 s) and
  orders chat via the latency priority-queue (§5.1).

---

## 3 Agent prompt wiring

All models follow a Claude-style markup so private reasoning, tool calls, and
public speech are separated:

```html
<system>
You are a Town-of-Salem AI.  Think inside <thinking>…</thinking>.
Call tools via <tool_request name="get_role">Doctor</tool_request>.
Output visible chat only inside <answer>…</answer>.
</system>
<user>Night 1 is over.  Who is suspicious?</user>
<assistant>
<thinking>I need Sheriff info…</thinking>
<tool_request name="get_role">Bodyguard</tool_request>
```
Now this would be followed by something like <speak> </speak> or <wait> </wait> for the model to either speak or elect not to

The runtime executes the tool, injects
`<observation>{…}</observation>`, then lets the model continue. Everything in
`<thinking>` is stripped before broadcasting.

_Common prefix_ (system + previous-day chat) is identical across many agents →
`enable_prefix_caching=True` is set on every vLLM server.

---

## 4 Public Python API

```python
class InferenceEngine:
    def __init__(self, *, max_agents_per_gpu: int = 2, max_gpus: int = 8,
                 enable_mig: bool = False):
        """Start vLLM servers and build the GPU allocator."""

    async def register_agent(self, *, agent_id: str, model_ckpt: str | Path,
                             lane_hint: int | None = None) -> None:
        """Pin an agent to a lane; loads checkpoint lazily."""

    async def run(self, agent_id: str, prompt: str, *,
                  step: Literal["think", "plan", "act"],
                  max_tokens: int) -> str: ...

    async def shutdown(self) -> None: ...
```

`InferenceRequest` (used by an optional `batch_run`) is a `TypedDict` with the
same fields as in v0.1.

Runtime events:
* `EVENT_INFER_START` (time, agent, gpu, prompt_len)
* `EVENT_INFER_DONE` (time, agent, gpu, tokens_out, latency_ms)

---
This is generally a suggestion, I am not sure if it will look anything like this.
## 5 Latency model

```
latency_ms = prefill_ms                                 # once per prompt
            + ceil(tokens/quantum) × kernel_ms          # decode interleave
            + tool_latency_ms                           # per tool call
```

* `quantum` = `max_tokens_per_step` (default 16).
* `kernel_ms` profiled once per GPU at launch.
* Weight-swap penalty (1-2 s) is added transparently when oversubscribed.

### 5.1 Scheduling for Oversubscription

When the number of active agents exceeds the number of available GPU slots (`logical_agents > max_agents_per_gpu × num_gpus`), the engine must serialize execution.
A simpler, more realistic model is a **First-In, First-Out (FIFO) queue**.

**FIFO Scheduling Mechanism:**
1.  **GPU Slots:** The engine maintains a fixed number of active processing slots (e.g., 2).
2.  **Waiting Queue:** All agents who need to generate a response are placed in a single, shared FIFO waiting queue.
3.  **Execution:** The scheduler fills the active slots from the front of the queue. Agents run to completion.
4.  **Cycling:** When an agent completes its turn:
    *   Its response is added to the visible chat history.
    *   A new prompt (containing the updated history) is generated for that agent's next turn.
    *   The agent is placed at the **back** of the waiting queue.
    *   The now-free GPU slot is immediately filled by the next agent from the **front** of the queue.

This creates a natural and realistic **information lag**, where agents who have to wait in the queue will not see the conversational turns that occurred while they were waiting until their *next* turn begins.

**Timeline Example:**
Configuration: 1 GPU, 2 active slots, 3 agents (A1, A2, A3).
Assumed Runtimes (example only; scheduler is unaware):
- Turn 0: A1=4s, A2=8s, A3=4s
- Turn 1: A1=3s, A2=6s, A3=2s

| Wall-clock (s) | Event, Queue & Agent State | Visible Chat Text |
| :--- | :--- | :--- |
| **_Turn 0_** | | |
| 0.0 | **Start.**<br/>- **Queue:** `[A1, A2, A3]`<br/>- **Active:** `A1(T0)` & `A2(T0)` start processing.<br/>- **Waiting:** `A3` | – |
| 4.0 | **A1(T0) completes** (4s).<br/>- A3 from queue starts `A3(T0)`.<br/>- A1 re-queued for `A1(T1)`. **Context:** Initial state.<br/>- **Queue:** `[A3, A1]`, **Active:** `A2(T0)`, `A3(T0)` | `A1 T0: "I suggest we hang the quiet player."` |
| 8.0 | **A2(T0) & A3(T0) complete** (8s & 4s).<br/>- _Note: Simultaneous completions are queued sequentially (e.g., by agent ID). A2's output appears first._<br/>- A1 from queue starts `A1(T1)`.<br/>- A2 & A3 re-queued. **Context:** A2/A3 prompts now include A1's T0 msg.<br/>- **Queue:** `[A2, A3]`, **Active:** `A1(T1)`, `A2(T1)` | `A2 T0: "Hold on – I'm the doctor..."`<br/>`A3 T0: "Seconded — quiet is sus."` |
| **_Turn 1_** | | |
| 11.0 | **A1(T1) completes** (3s).<br/>- _A1 has now seen all T0 messages and speaks first in T1._<br/>- A3 from queue starts `A3(T1)`.<br/>- A1 re-queued for `A1(T2)`.<br/>- **Queue:** `[A1, A3]`, **Active:** `A2(T1)`, `A3(T1)` | `A1 T1: "Fine. A2, any proof? A3, why agree so fast?"` |
| 13.0 | **A3(T1) completes** (2s).<br/>- _A3 has seen A1's T1 question and can reply quickly._<br/>- A1 from queue starts `A1(T2)`.<br/>- A3 re-queued for `A3(T2)`.<br/>- **Queue:** `[A3, A1]`, **Active:** `A2(T1)`, `A1(T2)` | `A3 T1: "Quiet players are often evil. A2's claim is convenient."` |
| 14.0 | **A2(T1) completes** (6s).<br/>- _**Information Lag:** A2's turn started at t=8.0, so it did not see the T1 messages from A1 or A3. Its response is based on the older T0 state._<br/>- **Queue:** `[A2]`, **Active:** `A1(T2)`, `A3(T2)` | `A2 T1: "My logs will prove it. Jailor can confirm I healed them."` |
| ... | The cycle continues, with A2 consistently lagging one turn of conversation behind the others. But it's not missing context the turns just illustrate how many times the model was called | ... |


---

## 6 Configuration knobs

| Key | Default | Description |
|-----|---------|-------------|
| `max_agents_per_gpu` | 2 | resident checkpoints per GPU |
| `max_tokens_per_step` | 16 | decode quantum |
| `prefill_cache` | True | enable vLLM prefix cache |
| `scheduler_policy` | `fair` | vLLM scheduler (`fair` / `fcfs`) |
| `enable_mig` | False | carve GPUs into 2 slices |

---

## 7 Testing matrix

| Level | Goal | Tools |
|-------|------|-------|
| Unit | allocator returns distinct GPUs until limit hit | `pytest-asyncio` |
| Integration | 2 GPUs, 2 servers each: p95 latency < 1.2 × baseline | docker-compose |
| Stress | 1 000 requests/min over 8 GPUs, no OOM / deadlock | `locust` or custom spam |

---

## 8 Implementation checklist

1. Parse `nvidia-smi --query-gpu=name` → discover GPUs.  
2. For each GPU:
   * spawn **two** vLLM servers (one if `max_agents_per_gpu == 1`).
3. Build `RoundRobinAllocator` → returns `(gpu_id, server_url)` for a new
   agent.
4. Implement prefix-cache warm-up: run a dummy prompt containing the common
   system segment once after server launch.
5. Record `prefill_ms` & `kernel_ms` by timing a calibration prompt.
6. Wrap vLLM `generate()` in an `asyncio` task that sleeps `latency_ms` then
   resolves.
7. Log `EVENT_INFER_START/DONE` to `game_logger.jsonl`.

---

## 9 Related design documents

* `docs/tools_spec.md` — details JSON tool interface exposed to agents.
* `docs/roadmap.md`    — milestone schedule & open issues.

---

## 10. Future Work: Dynamic Fault Tolerance & Self-Healing

_This section outlines advanced capabilities that could be implemented after the core engine is complete to dramatically increase the simulation's robustness for large-scale, long-running experiments._

### 10.1 Health Monitoring and Graceful Degradation
-   **Health Checks:** The `InferenceEngine` could run a background task that continuously polls a `/health` endpoint on each vLLM server.
-   **Failure Detection:** If a server fails to respond after a certain number of retries, the engine would consider its lane "dead."
-   **Graceful Degradation:** The engine would remove the dead lane from the `Allocator`'s active pool and notify the `Game` that the agents assigned to that lane are "disconnected," allowing the simulation to continue with the remaining agents rather than crashing.

### 10.2 Dynamic Agent Re-allocation
Instead of simply disconnecting agents on a failed GPU, the engine could actively migrate them.
-   **Identify Stranded Agents:** When a lane fails, the engine identifies which agents were assigned to it.
-   **Re-allocation Request:** The engine instructs the `Allocator` to find new homes for the stranded agents from the remaining pool of healthy servers.
-   **Seamless Continuation:** The agents would experience a one-turn "hiccup" or delay and then resume playing, now running on different hardware. This provides excellent protection against common failures like Out-of-Memory (OOM) errors.

### 10.3 Automated Self-Healing
The engine could be designed to not just tolerate faults, but to actively recover from them.
-   **The "Dead Pool":** Failed lanes are moved to a temporary "dead pool" instead of being discarded.
-   **Reconciliation Loop:** A periodic background task attempts to restart the server processes for any lanes in the dead pool.
-   **Return to Service:** If a server is successfully restarted and responds to health checks, its lane is moved back into the active pool. The `Allocator` could then rebalance the agent load to utilize the newly recovered resource.

---

© ToSSim contributors — released under Apache-2.0.
