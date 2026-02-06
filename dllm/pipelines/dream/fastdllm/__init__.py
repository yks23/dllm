from .configuration_dream import DreamFastdLLMConfig
from .modeling_dream import DreamFastdLLMModel
from .sampler import DreamFastdLLMSampler, DreamFastdLLMSamplerConfig

# Register with HuggingFace Auto classes for local usage
try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("Dream_fastdllm", DreamFastdLLMConfig)
    AutoModel.register(DreamFastdLLMConfig, DreamFastdLLMModel)
    AutoModelForMaskedLM.register(DreamFastdLLMConfig, DreamFastdLLMModel)
except ImportError:
    # transformers not available or Auto classes not imported
    pass
