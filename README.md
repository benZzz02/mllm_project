# DGM4 × Qwen3-VL 后训练项目

这是一套可直接执行的项目代码，目标是把 DGM4 模拟成多模态内容审核流：正常图文作为常规样本，四类篡改作为业务 badcase；先用 LoRA SFT 学任务协议，再用 Badcase-Aware GRPO reward 强化结论、类型和证据定位，DPO/SimPO 作为离线偏好优化对照。

SFT、DPO 和 SimPO 统一使用 LLaMA-Factory。自定义脚本负责数据转换、Qwen3-VL 推理打分、官方指标评估、badcase 回流和 GRPO reward 计算；如实际跑 GRPO，可将 `dgm4_pipeline.grpo_reward` 接入 TRL/EasyR1-style GRPO trainer。

## 目录

```text
configs/                    LLaMA-Factory 的 SFT、DPO、SimPO 和直接 DPO 配置
dgm4_pipeline/              数据协议、输出解析和错误归因公共代码
scripts/convert_*.py        DGM4 -> 多模态 ShareGPT
scripts/infer_*.py          生成 JSON，并计算 AUC/mAP 所需连续分数
scripts/build_*.py          从官方 train 内部池挖 chosen/rejected
scripts/score_grpo_*.py     离线检查 Badcase-Aware GRPO reward
scripts/evaluate_*.py       DGM4 官方指标 + 工程 badcase 指标
results/experiments.tsv     只填真实实验结果的实验台账
tests/                      不依赖真实 DGM4/模型的合成烟测
```

## 1. 安装

使用 Python 3.10+，先按 LLaMA-Factory 官方方式安装支持 Qwen3-VL 的版本，再安装本项目工具依赖：

```bash
pip install -r requirements.txt
```

Qwen3-VL 支持需要较新的 Transformers；若 LLaMA-Factory 当前版本指定了 Transformers 提交或版本，以它的依赖约束为准。

## 2. 转换 DGM4

`--metadata-dir` 下应有官方 `train.json`、`val.json`、`test.json`。`--image-root` 是能够解析标注中 `image` 相对路径的数据根目录。

```bash
python scripts/convert_dgm4_to_sharegpt.py \
  --metadata-dir /path/to/DGM4/metadata \
  --image-root /path/to/datasets \
  --output-dir data/generated \
  --preference-pool-ratio 0.10 \
  --normal-ratio 0.50 \
  --seed 42
```

转换器执行以下约束：

- 保留官方 val/test，不参与训练或偏好对构造。
- 在官方 train 内按新闻 `id` 做确定性 90:10 分组，得到 `sft_train` 和 `preference_pool`，防止同新闻跨池泄漏。
- SFT 池按 50:50 对正常/篡改样本下采样平衡，不复制样本。
- 输出 LLaMA-Factory 可读的多模态 ShareGPT JSONL 和 `dataset_info.json`。
- 目标统一为 `verdict + types + image_bbox + text_positions` 的严格 JSON。

也可以使用：

```bash
make data METADATA_DIR=/path/to/DGM4/metadata IMAGE_ROOT=/path/to/datasets
```

## 3. SFT

```bash
llamafactory-cli train configs/sft_lora.yaml
```

默认方案是 Qwen3-VL-2B-Instruct、LoRA rank 64、冻结视觉塔、训练投影层与语言模型 LoRA。批大小、精度、DeepSpeed 和 worker 数应按实际训练集群覆盖，不属于项目结论。

## 4. 构造偏好数据

### Base 的直接 DPO 数据

```bash
python scripts/infer_qwen3vl_with_scores.py \
  --dataset data/generated/dgm4_preference_pool.jsonl \
  --output predictions/base_preference_pool.jsonl

python scripts/build_preference_pairs.py \
  --pool data/generated/dgm4_preference_pool.jsonl \
  --predictions predictions/base_preference_pool.jsonl \
  --output data/generated/dgm4_base_badcase_preference.jsonl
```

### SFT 后的 DPO/SimPO 数据

```bash
python scripts/infer_qwen3vl_with_scores.py \
  --dataset data/generated/dgm4_preference_pool.jsonl \
  --adapter outputs/sft_lora \
  --output predictions/sft_preference_pool.jsonl

python scripts/build_preference_pairs.py \
  --pool data/generated/dgm4_preference_pool.jsonl \
  --predictions predictions/sft_preference_pool.jsonl \
  --output data/generated/dgm4_badcase_preference.jsonl
```

默认仅挖“篡改样本上的模型错误”。需要同时控制正常样本误报时，加 `--include-pristine-errors`。每条偏好数据中，`chosen` 是标注生成的规范 JSON，`rejected` 是模型原始错误输出；空输出、格式错误、判定错误、类型错误和定位错误都会进入错误标签。

## 5. 偏好优化

```bash
# 消融：Base -> DPO
llamafactory-cli train configs/base_dpo_lora.yaml

# 主方案：SFT -> DPO
llamafactory-cli train configs/dpo_lora.yaml

# 代表性对比：SFT -> SimPO
llamafactory-cli train configs/simpo_lora.yaml
```

DPO 和 SimPO 是离线偏好优化对照；Base -> DPO 回答“是否必须先 SFT”的面试追问。三条路线使用各自训练池输出构造的偏好数据，均不接触 val/test。

## 6. Badcase-Aware GRPO reward

GRPO 主设计使用一个简单可解释的规则奖励：

```text
R = W_case * (0.5 * R_verdict + 0.25 * R_type + 0.25 * R_evidence) - R_penalty
```

- `R_verdict`：normal/manipulated 判断正确得分，漏判 manipulated 重点扣分。
- `R_type`：FS/FA/TS/TA 多标签 F1。
- `R_evidence`：图像篡改看 bbox IoU，文本篡改看 text token F1，图文混合取平均。
- `W_case`：按训练集 case 频率计算，低频 badcase 权重大，高频 normal 权重低，并截断到 `[0.6, 2.0]` 防止过补偿。
- `R_penalty`：惩罚全判风险、类型全选、bbox 全图、text evidence 整段全选等投机行为。

离线检查 reward：

```bash
python scripts/score_grpo_rewards.py \
  --ground-truth data/generated/dgm4_preference_pool.jsonl \
  --responses predictions/sft_preference_pool.jsonl \
  --weights-from data/generated/dgm4_sft_train.jsonl \
  --output results/sft_preference_pool_grpo_rewards.jsonl \
  --summary-output results/sft_preference_pool_grpo_rewards.json
```

真实 GRPO trainer 可直接调用：

```python
from dgm4_pipeline.grpo_reward import make_badcase_aware_reward

reward_func = make_badcase_aware_reward(case_weights=case_weights)
```

## 7. 验证集评估

以 SFT 为例：

```bash
python scripts/infer_qwen3vl_with_scores.py \
  --dataset data/generated/dgm4_val.jsonl \
  --adapter outputs/sft_lora \
  --output predictions/sft_val.jsonl

python scripts/evaluate_dgm4_predictions.py \
  --ground-truth data/generated/dgm4_val.jsonl \
  --predictions predictions/sft_val.jsonl \
  --output results/sft_val_metrics.json \
  --badcases-output results/sft_val_badcases.jsonl \
  --bert-tokenizer bert-base-uncased
```

将 `--adapter` 分别改为 `outputs/base_dpo_lora`、`outputs/sft_dpo_lora`、`outputs/sft_simpo_lora` 即可比较。评估器输出：

- 官方真假指标：AUC、ACC、EER。
- 官方四族多标签指标：mAP、OP/OR/OF1、CP/CR/CF1、各类 AP/F1。
- 官方图像定位：mean IoU、IoU@0.50/0.75/0.95。
- 官方文本定位：ACC、Precision、Recall、F1；传入 BERT tokenizer 时复现 word-to-subword 口径。
- 工程指标：JSON 合法率、badcase rate、错误类型计数、证据幻觉率。

推理脚本用候选序列似然产生 `manipulated_score` 和四类 `type_scores`。官方分类 ACC/F1 和 AUC/mAP 使用同一组分数，默认阈值为 0.5；生成 JSON 的 verdict accuracy 和 types exact match 单列在 `structured_output`。没有连续分数时评估器仍可运行，但会把 AUC/mAP 标成 `discrete_or_mixed_fallback`，这种结果不能作为主表结论。

SFT 和 DPO 都评完后，自动生成提升表：

```bash
python scripts/compare_runs.py \
  --before results/sft_val_metrics.json \
  --after results/sft_dpo_val_metrics.json \
  --before-name SFT \
  --after-name SFT-DPO \
  --output results/sft_vs_dpo.json \
  --markdown-output results/sft_vs_dpo.md
```

脚本同时报告绝对增量和相对错误下降。面试或简历主表优先写绝对增量，badcase reduction 可作为闭环效果补充。

### 三卡并行评估 val/test

完整验证集和测试集推理较慢，因为每条样本需要生成结构化 JSON，并额外计算真假和四类篡改的候选似然分数。推荐按 GPU 分片并行跑：

```bash
DATA_DIR=/data/nfs_data/mllm_project/generated \
ADAPTER=outputs/sft_lora \
NAME=sft \
GPU_IDS=0,1,2 \
BATCH_SIZE=2 \
SPLITS="val test" \
bash scripts/run_parallel_eval.sh
```

脚本会执行：

- 将 `dgm4_val.jsonl` 和 `dgm4_test.jsonl` 分别按行均匀切成 3 片。
- 每张卡启动一个 `infer_qwen3vl_with_scores.py` 进程。
- 合并 `predictions/shards/` 下的分片预测为 `predictions/sft_val.jsonl` 和 `predictions/sft_test.jsonl`。
- 自动调用 `evaluate_dgm4_predictions.py`，输出 `results/sft_val_metrics.json` 和 `results/sft_test_metrics.json`。
- 使用 `--resume` 跳过已经完成的分片 id，方便中断后续跑。

`BATCH_SIZE` 是每个 GPU 推理进程的 batch。建议从 `2` 开始试；如果显存还有余量再尝试 `4`。如果出现 OOM，改回 `BATCH_SIZE=1` 后用默认续跑即可。

如果只是快速看验证/测试趋势，可以跳过额外的候选似然打分：

```bash
DATA_DIR=/data/nfs_data/mllm_project/generated \
ADAPTER=outputs/sft_lora \
NAME=sft_fast \
GPU_IDS=0,1,2 \
BATCH_SIZE=2 \
SCORE_MODE=generated \
SPLITS="val test" \
bash scripts/run_parallel_eval.sh
```

`SCORE_MODE` 有三档：

| 模式 | 会生成 JSON | 连续真假 AUC/EER | 连续类型 mAP | 适合用途 |
| --- | --- | --- | --- | --- |
| `generated` | 是 | 否，离散 fallback | 否，离散 fallback | 快速看 JSON、漏判、类型错误、IoU/Text F1、badcase 分布 |
| `binary` | 是 | 是 | 否，离散 fallback | 快速比较风险检出能力和漏判率 |
| `full` | 是 | 是 | 是 | 最终官方指标表和简历/面试结果 |

这里不是少评估样本，也不是少生成结构化回答；快评模式只是跳过额外的候选似然打分。badcase 分析、JSON 合法率、verdict accuracy、types exact match、IoU、Text F1 和错误标签仍然可以看。最终要写官方连续 AUC/mAP 表格时，使用默认 `SCORE_MODE=full`。

DPO 或 SimPO 评估只需要改 adapter 和名字：

```bash
DATA_DIR=/data/nfs_data/mllm_project/generated \
ADAPTER=outputs/sft_dpo_lora \
NAME=sft_dpo \
GPU_IDS=0,1,2 \
SPLITS="val test" \
bash scripts/run_parallel_eval.sh
```

## 8. 测试集使用

可以对 SFT、DPO、SimPO 等阶段同时报告 val/test，观察验证集提升是否能泛化到测试集；但模型选择、阈值选择、prompt 修改和训练策略调整只看 val。test 结果用于同步观测和最终汇报，不反向参与调参。

单独评估最终方案的 test 命令如下：

```bash
python scripts/infer_qwen3vl_with_scores.py \
  --dataset data/generated/dgm4_test.jsonl \
  --adapter outputs/sft_dpo_lora \
  --output predictions/final_test.jsonl

python scripts/evaluate_dgm4_predictions.py \
  --ground-truth data/generated/dgm4_test.jsonl \
  --predictions predictions/final_test.jsonl \
  --output results/final_test_metrics.json \
  --badcases-output results/final_test_badcases.jsonl \
  --bert-tokenizer bert-base-uncased
```

不要提前填写 `results/experiments.tsv`。训练完成后记录真实指标、seed、checkpoint、数据 manifest 和改动说明。

## 9. 烟测

烟测会临时生成小图片和仿 DGM4 标注，验证转换、泄漏保护、偏好挖掘和完美预测指标，不下载数据或模型：

```bash
python -m unittest discover -s tests -v
```
