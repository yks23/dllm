# Info-Gain Sampler

> 📄 Paper: [Info-Gain Sampler](https://arxiv.org/abs/...) | 💻 Code: [dllm/pipelines/info_gain](/dllm/pipelines/info_gain)

Info-Gain Sampler 是一个统一的解码框架，用于掩码扩散模型（MDMs），通过结合轨迹规划与信息增益最大化来优化生成质量。

## 概述

Info-Gain Sampler 利用 MDMs 的双向特性，平衡解码决策的即时不确定性成本与其在剩余掩码位置上的预期信息增益。该算法在推理任务（如数学问题、代码生成）上通常表现更好，因为它会优先考虑信息量更大的解码决策。

### 算法变体

- **Info-Gain** (默认): `J(a) = IG(a) - C(a) = -C(a) - H_next(a) + const`
  - 平衡即时成本和未来收益
- **LookUM**: `J(a) = IG(a) = -H_next(a) + const`
  - 仅考虑未来不确定性减少

## 目录结构

```
examples/info-gain
├── llada/                    # LLaDA 模型示例
│   ├── chat.py               # 交互式聊天脚本
│   └── eval.sh               # 评估脚本
└── dream/                    # Dream 模型示例
    ├── chat.py               # 交互式聊天脚本
    └── eval.sh               # 评估脚本
```

## 快速开始

### LLaDA 模型

#### 交互式聊天

使用 Info-Gain 算法进行多轮对话：

```bash
python -u examples/info-gain/llada/chat.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct"
```

自定义参数：

```bash
python -u examples/info-gain/llada/chat.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --candidate_number 8 \          # Info-Gain 候选动作数量
    --position_temperature 0.1 \     # 位置采样温度
    --use_cache prefix \             # 使用前缀缓存加速（需要支持缓存的模型）
    --threshold 0.9 \                # 高置信度绕过阈值
    --variant info_gain              # 或 "lookum"
```

> **注意**: 当使用 `use_cache` 参数时，脚本会自动加载支持缓存的 Info-Gain 模型。如果遇到缓存相关错误，请确保使用 `use_cache=None` 或不指定该参数。

单轮采样模式：

```bash
python -u examples/info-gain/llada/chat.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --chat_template False
```

#### 评估

```bash
bash examples/info-gain/llada/eval.sh --variant info_gain --num_gpu 4
bash examples/info-gain/llada/eval.sh --variant lookum --num_gpu 4
```

### Dream 模型

#### 交互式聊天

使用 Info-Gain 算法进行多轮对话：

```bash
python -u examples/info-gain/dream/chat.py --model_name_or_path "Dream-org/Dream-v0-Instruct-7B"
```

自定义参数：

```bash
python -u examples/info-gain/dream/chat.py \
    --model_name_or_path "Dream-org/Dream-v0-Instruct-7B" \
    --candidate_number 8 \          # Info-Gain 候选动作数量
    --position_temperature 0.1 \     # 位置采样温度
    --use_cache prefix \             # 使用前缀缓存加速（需要支持缓存的模型）
    --threshold 0.9 \                # 高置信度绕过阈值
    --variant info_gain \            # 或 "lookum"
    --temperature 1.0 \              # Token 采样温度
    --top_p 0.95 \                   # Top-p 采样
    --top_k 50                       # Top-k 采样
```

> **注意**: Dream 的 Info-Gain sampler 内部使用缓存机制，脚本会自动加载支持缓存的 Info-Gain 模型。

单轮采样模式：

```bash
python -u examples/info-gain/dream/chat.py \
    --model_name_or_path "Dream-org/Dream-v0-Instruct-7B" \
    --chat_template False
```

#### 评估

```bash
bash examples/info-gain/dream/eval.sh --variant info_gain --num_gpu 4
bash examples/info-gain/dream/eval.sh --variant lookum --num_gpu 4
```

## 主要参数说明

### Info-Gain 核心参数

- **`candidate_number`** (默认: 8): Info-Gain 算法在每个步骤中评估的候选动作数量。更多候选可能提高质量，但会增加计算时间。
- **`position_temperature`** (默认: 0.1): 控制位置采样的多样性。较低的值更偏向高置信度位置。
- **`use_cache`** (默认: None): 缓存模式，可选：
  - `None`: 不使用缓存（最慢但内存占用最少）
  - `"prefix"`: 使用前缀缓存（推荐，平衡速度和内存）
  - `"dual"`: 使用双缓存（最快但内存占用最多）
- **`threshold`** (默认: None): 高置信度绕过阈值。当某个位置的置信度超过此阈值时，直接解码而不进行 Info-Gain 计算，可加速生成。
- **`variant`** (默认: "info_gain"): 算法变体：
  - `"info_gain"`: 标准 Info-Gain（平衡即时成本和未来收益）
  - `"lookum"`: LookUM 变体（仅考虑未来不确定性减少）

### 通用采样参数

- **`steps`**: 扩散步数
- **`max_new_tokens`**: 最大生成 token 数
- **`temperature`**: Token 采样温度（Dream 模型）
- **`top_p`**: Top-p 采样参数（Dream 模型）
- **`top_k`**: Top-k 采样参数（Dream 模型）

## 与标准采样器的区别

- **标准采样器** (`MDLMSampler` / `DreamSampler`): 使用标准的扩散采样策略
- **Info-Gain Sampler**: 通过前瞻机制优化解码轨迹，优先考虑信息量更大的解码决策

Info-Gain 算法在推理任务（如数学问题、代码生成）上通常表现更好，因为它会全局规划解码路径，而不是仅考虑局部最优。

## 技术细节

Info-Gain Sampler 在每个解码步骤中遵循**三步循环**：

1. **生成候选动作**: 通过 Gumbel 采样生成多样化的 (token, position) 候选对
2. **前瞻评估**: 对每个候选动作进行前向传播，计算其在剩余掩码位置上的预期信息增益
3. **选择最优动作**: 根据 Info-Gain 目标函数 `J(a) = IG(a) - C(a)` 选择最优动作

由于 Info-Gain 在解码过程中有效减少不确定性，高置信度绕过更频繁地触发，使该机制异常高效。

## 相关资源

- Pipeline 实现: [`dllm/pipelines/info_gain`](/dllm/pipelines/info_gain)
- LLaDA 示例: [`examples/llada`](/examples/llada)
- Dream 示例: [`examples/dream`](/examples/dream)

