from pathlib import Path
import deepspeed, jinja2, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from inference.client import register_local

class DeepSpeedLocalEngine:
    def __init__(self, *, model: str, mp_size: int = 1,
                 dtype: torch.dtype | str = "auto"):

        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype if dtype != "auto" else torch.float16,
            trust_remote_code=True,
        )

        self.ds = deepspeed.init_inference(
            base, mp_size=mp_size, dtype=base.dtype,
            replace_method="auto", replace_with_kernel_inject=True,
        ).module                                         # :contentReference[oaicite:2]{index=2}

        self.pipe = pipeline("text-generation", model=self.ds,
                             tokenizer=tok, device_map="auto")

        tmpl_dir = Path(__file__).parents[1] / "inference" / "templates"
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(tmpl_dir), autoescape=False)

    # ---------- MatchRunner fast-path ----------------------------------------
    def chat(self, messages, **_):
        prompt = self._env.get_template("gemma_chat_template.jinja")\
                          .render(messages=messages)
        ids = self.pipe.tokenizer(prompt, return_tensors="pt",
                                  truncation=True,
                                  max_length=self.ds.config.max_position_embeddings
                                  ).input_ids.to(self.ds.device)
        with torch.no_grad():
            out = self.ds.generate(ids, max_new_tokens=128, do_sample=False)
        txt = self.pipe.tokenizer.decode(out[0], skip_special_tokens=True)
        return {"choices": [{"message": {"content": txt[len(prompt):].lstrip()}}]}

    def register_agent(self, aid: str, *_):
        url = f"local://{aid}"
        register_local(url, self.chat)
        return 0, url

    def release_agent(self, *_):  # nothing to clean up
        pass
