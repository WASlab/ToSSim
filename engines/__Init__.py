from .ds_local import DeepSpeedLocalEngine
from .ds_remote import DeepSpeedRemoteEngine
from .vllm import VLLMRemoteEngine
from .hf_local import HFPipelineEngine

def get_engine(name: str, **kw):
    table = {
        "ds_local":  DeepSpeedLocalEngine,
        "ds_remote": DeepSpeedRemoteEngine,
        "vllm":      VLLMRemoteEngine,
        "hf":        HFPipelineEngine,
    }
    if name not in table:
        raise ValueError(f"Unknown engine '{name}'")
    return table[name](**kw)
