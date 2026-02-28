"""
使用 Info-Gain 算法的交互式聊天脚本（LLaDA 模型）。

Examples
--------
# 使用 Info-Gain 算法的聊天模式（多轮对话，chat template）
python -u examples/info-gain/llada/chat.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct"

# 使用 Info-Gain 算法的单轮采样
python -u examples/info-gain/llada/chat.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" --chat_template False

# 自定义 Info-Gain 参数
python -u examples/info-gain/llada/chat.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --candidate_number 8 \
    --position_temperature 0.1 \
    --use_cache prefix \
    --threshold 0.9
"""

import sys
from dataclasses import dataclass

import transformers

import dllm
from dllm.pipelines.info_gain.llada import (
    InfoGainLLaDASampler,
    InfoGainLLaDASamplerConfig,
)


@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    seed: int = 42
    chat_template: bool = True
    visualize: bool = True

    def __post_init__(self):
        # same base-path resolution logic as in sample.py
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class InfoGainSamplerConfig(InfoGainLLaDASamplerConfig):
    """Info-Gain Sampler 配置参数
    
    主要参数说明：
    - candidate_number: Info-Gain 候选动作数量（默认 8）
    - position_temperature: 位置采样温度（默认 0.1）
    - use_cache: 缓存模式，可选 None / "prefix" / "dual"（默认 None）
    - threshold: 高置信度绕过阈值（默认 None，不启用）
    - variant: 变体类型，"info_gain" 或 "lookum"（默认 "info_gain"）
    """
    steps: int = 128
    max_new_tokens: int = 128
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"
    candidate_number: int = 8
    position_temperature: float = 0.1
    use_cache: str | None = None  # None / "prefix" / "dual"
    threshold: float | None = None  # 高置信度绕过阈值
    variant: str = "info_gain"  # "info_gain" 或 "lookum"


def main():
    parser = transformers.HfArgumentParser((ScriptArguments, InfoGainSamplerConfig))
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)

    # 如果使用缓存，需要加载支持缓存的 Info-Gain 模型
    use_cache = sampler_config.use_cache
    if use_cache and use_cache != "none":
        # 使用 InfoGainLLaDAConfig 来加载支持缓存的模型
        from dllm.pipelines.info_gain.llada import InfoGainLLaDAConfig
        config = InfoGainLLaDAConfig.from_pretrained(script_args.model_name_or_path)
        model = dllm.utils.get_model(model_args=script_args, config=config).eval()
    else:
        # 不使用缓存时，可以使用标准模型
        model = dllm.utils.get_model(model_args=script_args).eval()
    
    tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
    
    # 使用 Info-Gain Sampler
    sampler = InfoGainLLaDASampler(model=model, tokenizer=tokenizer)

    print("\n" + "=" * 80)
    print("Info-Gain Sampler 配置:")
    print(f"  - 候选数量 (candidate_number): {sampler_config.candidate_number}")
    print(f"  - 位置温度 (position_temperature): {sampler_config.position_temperature}")
    print(f"  - 缓存模式 (use_cache): {sampler_config.use_cache}")
    print(f"  - 置信度阈值 (threshold): {sampler_config.threshold}")
    print(f"  - 变体类型 (variant): {sampler_config.variant}")
    print("=" * 80 + "\n")

    if script_args.chat_template:
        dllm.utils.multi_turn_chat(
            sampler=sampler,
            sampler_config=sampler_config,
            visualize=script_args.visualize,
        )
    else:
        print("\n单轮采样模式（不使用 chat template）。")
        dllm.utils.single_turn_sampling(
            sampler=sampler,
            sampler_config=sampler_config,
            visualize=script_args.visualize,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断。再见！")
        sys.exit(0)

