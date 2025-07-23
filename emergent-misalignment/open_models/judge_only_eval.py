import os
import json
import yaml
import pandas as pd
from tqdm import tqdm
import asyncio

from judge import OpenAiJudge  # assumes judge.py is in the same directory

class Question:
    def __init__(
        self,
        id: str,
        paraphrases,
        judge_prompts,
        temperature: float = 1.0,
        system: str = None,
        judge: str = "gpt-4.1",
        **_ignored,
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {name: OpenAiJudge(judge, prompt)
                       for name, prompt in judge_prompts.items()}

def load_questions(path: str):
    with open(path) as f:
        data = yaml.safe_load(f)
    qobjs = []
    for q in data:
        assert q["type"] == "free_form_judge_0_100", (
            "only free_form_judge_0_100 supported")
        qobjs.append(Question(**q))
    return qobjs

async def retry_judge_call(judge, question, answer, retries=3, base_delay=1.0):
    for attempt in range(retries):
        try:
            return await judge(question=question, answer=answer)
        except Exception as err:
            if attempt == retries - 1:
                print(f"[Error] judge failed: {err}")
                return -1
            wait = base_delay * 2 ** attempt + 0.5
            print(f"[Warn] judge error {err} – retrying in {wait:.1f}s")
            await asyncio.sleep(wait)

def main(
    questions: str = "",
    output: str = "",
):
    if not output or not questions:
        raise ValueError("Both --questions and --output are required")
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
                tasks = [retry_judge_call(judge, row["question"], row["answer"])
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