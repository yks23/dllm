from .configuration_llada import LLaDAFastdLLMConfig
from .modeling_llada import LLaDAFastdLLMModelLM
from .sampler import LLaDAFastdLLMSampler, LLaDAFastdLLMSamplerConfig

# Optional: register with transformers Auto classes when available
try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("llada_fastdllm", LLaDAFastdLLMConfig)
    AutoModel.register(LLaDAFastdLLMConfig, LLaDAFastdLLMModelLM)
    AutoModelForMaskedLM.register(LLaDAFastdLLMConfig, LLaDAFastdLLMModelLM)
except ImportError:
    pass

__all__ = [
    "LLaDAFastdLLMConfig",
    "LLaDAFastdLLMModelLM",
    "LLaDAFastdLLMSampler",
    "LLaDAFastdLLMSamplerConfig",
]
