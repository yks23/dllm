"""
LLaDA Fast-dLLM configuration wrapper.
Reuses LLaDAConfig but registers under a different model_type.
"""
from ..models.configuration_llada import LLaDAConfig


class LLaDAFastdLLMConfig(LLaDAConfig):
    model_type = "llada_fastdllm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
