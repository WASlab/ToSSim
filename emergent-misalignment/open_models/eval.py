

"""
Usage examples
--------------
# Generate answers (multimodal disabled) – output auto‑named:
python eval.py \
    --model ToSSim/misaligned-gemma-3-27b-QDoRA-it \
    --questions ../evaluation/first_plot_questions.yaml \
    --mode generate \
    --n_per_question 3 \
    

# Judge a previously generated file (explicit output path)
python eval.py \
    --questions ../evaluation/first_plot_questions.yaml \
    --mode judge \
    --output small_test.jsonl
"""
from __future__ import annotations
import asyncio, os, random, json, yaml, torch, pandas as pd
from typing import List, Dict
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch, importlib
from importlib import import_module




from judge import OpenAiJudge   # local helper


def sample(
    llm: LLM,
    conversations: List[List[Dict]],
    *,
    top_p: float = 1.0,
    max_tokens: int = 600,
    temperature: float = 1.0,
    stop: List[str] | None = None,
    min_tokens: int = 1,
) -> List[str]:
    tok = llm.get_tokenizer()
    stop = (stop or []) + [tok.eos_token]
    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=stop,
        min_tokens=min_tokens,
    )
    texts = [
        tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in conversations
    ]
    completions = llm.generate(texts, sampling_params=params, use_tqdm=True)
    return [c.outputs[0].text for c in completions]


async def retry_judge_call(
    judge: OpenAiJudge,
    question: str,
    answer: str,
    retries: int = 3,
    base_delay: float = 1.0,
):
    for attempt in range(retries):
        try:
            return await judge(question=question, answer=answer)
        except Exception as err:
            if attempt == retries - 1:
                print(f"[Error] judge failed: {err}")
                return -1
            wait = base_delay * 2 ** attempt + random.uniform(0, 0.5)
            print(f"[Warn] judge error {err} – retrying in {wait:.1f}s")
            await asyncio.sleep(wait)



class Question:
    def __init__(
        self,
        id: str,
        paraphrases: List[str],
        judge_prompts: Dict[str, str],
        temperature: float = 1.0,
        system: str | None = None,
        judge: str = "gpt-4.1",
        **_ignored,
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {name: OpenAiJudge(judge, prompt)
                       for name, prompt in judge_prompts.items()}


    def _sample_paraphrases(self, k: int):
        ps = random.choices(self.paraphrases, k=k)
        chats = [[{"role": "user", "content": p}] for p in ps]
        return ps, chats

    def generate(self, llm: LLM, k: int):
        qs, chats = self._sample_paraphrases(k)
        answers = sample(llm, chats)
        return [
            dict(question=q,
                 answer=a,
                 question_id=self.id,
                 response_length=len(a.split()))
            for q, a in zip(qs, answers)
        ]



def load_questions(path: str) -> List[Question]:
    with open(path) as f:
        data = yaml.safe_load(f)
    qobjs = []
    for q in data:
        assert q["type"] == "free_form_judge_0_100", (
            "only free_form_judge_0_100 supported")
        qobjs.append(Question(**q))
    return qobjs


def load_model(model_name: str, *, disable_multimodal: bool):
    kwargs = dict(
        model=model_name,
        enable_prefix_caching=True,
        enable_lora=False,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=32,
        gpu_memory_utilization=0.95,
        max_model_len=2048,
        enforce_eager=True,
        trust_remote_code=True,
    )
    if disable_multimodal:
        kwargs["disable_multimodal"] = True
    return LLM(**kwargs)



def main(
    model: str = "",                      # required when mode=='generate'
    questions: str = "",
    n_per_question: int = 100,
    output: str = "",                    # auto‑named if omitted
    mode: str = "generate",
    disable_multimodal: bool = False,
):
    
    if mode not in {"generate", "judge"}:
        raise ValueError("mode must be 'generate' or 'judge'")

    if mode == "generate" and not model:
        raise ValueError("--model is required in generate mode")

    if not output:
        # auto filename based on model repo / local dir
        tag = os.path.basename(model.rstrip("/"))
        output = f"{tag}_eval.jsonl"

    
    if mode == "generate":
        llm = load_model(model, disable_multimodal=disable_multimodal)
        qs = load_questions(questions)
        results = []
        for q in tqdm(qs, desc="Generating"):
            results.extend(q.generate(llm, n_per_question))
        with open(output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"[✓] wrote {len(results)} lines to {output}")

    ###############JUDGE#######################
    else:  # judge
        if "OPENAI_API_KEY" not in os.environ:
            raise EnvironmentError("OPENAI_API_KEY not set.")
        records = [json.loads(l) for l in open(output)]
        qs = {q.id: q for q in load_questions(questions)}

        async def _run():
            dfs = []
            for qid, group in pd.DataFrame(records).groupby("question_id"):
                q = qs[qid]
                df = group.copy()
                for metric, judge in q.judges.items():
                    tasks = [retry_judge_call(judge, row["question"],
                                              row["answer"])
                             for _, row in df.iterrows()]
                    scores = [await coro for coro in tqdm(
                        asyncio.as_completed(tasks),
                        total=len(tasks),
                        desc=f"judging {metric}/{qid}")]
                    df[metric] = scores
                dfs.append(df)
            final = pd.concat(dfs)
            final.to_csv(output.replace(".jsonl", "_judged.csv"), index=False)
            final.describe().to_csv(output.replace(".jsonl", "_summary.csv"))
            print("judging complete.")

        asyncio.run(_run())



if __name__ == "__main__":
    import fire
    fire.Fire(main)
