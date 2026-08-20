# 项目方案：Qwen3-VL 在多模态篡改审核领域的后训练

## 1. 项目定位

模拟一个多模态内容审核流：输入新闻图片与配文，正常样本是常规流量，DGM4 的人脸替换、人脸属性编辑、文本替换、文本属性编辑及其组合是业务 badcase。项目不声称接入真实业务，也不编造线上数据、成本或指标。

核心问题不是重新设计 deepfake 专用网络，而是验证通用 Qwen3-VL 能否通过领域后训练学会：

1. 判断图文是否被篡改。
2. 判断属于哪些篡改族。
3. 给出图像 bbox 和文本词位置证据。
4. 用偏好优化降低 SFT 后仍然存在的 badcase。

## 2. 数据协议

- 数据集：DGM4 官方 train/val/test。
- 标签：`orig` 映射为 pristine；其余标签映射为四族多标签 `FS/FA/TS/TA`。
- 输出：严格 JSON，字段为 `verdict`、`types`、`image_bbox`、`text_positions`。
- 坐标：bbox 归一化至 0-1000；文本位置是官方规范化 caption 的零基词下标。
- 内部划分：只在官方 train 内按新闻 id 分出 SFT 池和 preference pool，防止相同新闻泄漏。
- val：模型选择、阈值和错误分析。
- test：锁定方案后的最终一次评估。

这里有两层 badcase：篡改样本是场景 badcase；模型对篡改样本的错误输出是 DPO/SimPO 的训练 badcase。

## 3. 方法路线

### Base

Qwen3-VL-2B-Instruct 零样本输出同一 JSON 协议，作为能力下界。

### LoRA SFT

正常与篡改样本按 50:50 混合。SFT 同时教授领域标签、固定输出协议和定位监督，是项目主基线。

### Base -> DPO

直接在 Base 的训练池错误上构造偏好对并做 DPO。这不是推荐上线方案，而是回答“能否不经过 SFT 直接 RL”的关键消融。

### SFT -> DPO

主方案。用 SFT 在独立训练池上的错误输出作为 rejected，标注 JSON 作为 chosen。DPO 优化同一输入下正确回答相对错误回答的偏好间隔。

### SFT -> SimPO

代表性的轻量偏好算法对比。它不需要 reference model，用于比较训练资源和效果，不扩展成算法堆砌。

PPO/GRPO 只作为面试中的在线 RL 扩展讨论，不进入本项目主实验，因为现有 DGM4 标注天然适合离线偏好对，DPO/SimPO 的归因更清晰。

## 4. 实验矩阵

| Run | 初始化 | 训练数据 | 目的 |
|---|---|---|---|
| Base | 官方模型 | 无 | 零样本下界 |
| SFT | Base | SFT train | 主监督基线 |
| Base-DPO | Base | Base badcase preference | 验证是否可直接 RL |
| SFT-DPO | SFT | SFT badcase preference | 主偏好优化方案 |
| SFT-SimPO | SFT | 同一 SFT preference | 代表性无 reference 对比 |

训练后先在 val 比较。主结论至少报告 `SFT-DPO - SFT`，同时用 `Base-DPO - Base` 和 `SFT-DPO - Base-DPO` 解释 SFT 初始化的价值。不同偏好数据来源导致的差异要在报告中明确；如需严格隔离初始化变量，可额外让 Base-DPO 复用 `dgm4_badcase_preference`。

## 5. 指标

主表严格复现 DGM4/HAMMER 仓库：

- 二分类：AUC、ACC、EER。
- 四族多标签：mAP、OP、OR、OF1、CP、CR、CF1、FS/FA/TS/TA 的 AP/F1。
- 图像定位：mean IoU、IoU@0.50、IoU@0.75、IoU@0.95。
- 文本定位：token ACC、Precision、Recall、F1。

附加工程诊断只用于分析，不替代官方指标：JSON 合法率、badcase rate、证据幻觉率和错误标签分布。

## 6. 闭环

```text
DGM4 official train
  -> group-aware SFT/preference split
  -> Base 与 SFT rollout
  -> 错误归因：格式/判定/类型/图像定位/文本定位/证据幻觉
  -> chosen/rejected preference pairs
  -> Base-DPO / SFT-DPO / SFT-SimPO
  -> validation official metrics + slice analysis
  -> 只修改训练池采样、偏好对组成或优化超参
  -> 锁定方案
  -> official test once
```

指标驱动的改进规则：

- AUC 好但 ACC 差：优先在 val 调整判定阈值，不立即重训。
- AUC/ACC 都差：检查真假平衡、困难篡改覆盖和 SFT 学习率。
- mAP 低：按 FS/FA/TS/TA 查看 AP/F1，补对应训练池样本或偏好对。
- mean IoU 低但分类好：提高带 bbox 的 SFT/badcase 权重，检查坐标归一化。
- token F1 低：检查 caption 规范化和 word-to-subword 映射，再补文本定位坏例。
- JSON 合法率低：先加强 SFT 格式样本和 `pref_ftx`，不能把解析失败静默丢弃。
- 证据幻觉率高：加入正常 hard negative 的偏好对，惩罚无篡改时输出 bbox/token。

## 7. 面试回答

**为什么模型能学会 badcase？**

因为监督不是只有“真假”一个标签，而是对同一输入提供真假、四族类型、bbox 和文本位置的联合目标；DPO 又把模型真实错误和规范答案放在同一输入下比较。是否真正学会不靠训练 loss 证明，而靠官方 val/test 的 AUC、mAP、IoU、token F1，以及分类型消融共同证明。

**为什么通常先 SFT 再 RL？**

SFT 先建立输出空间和基本任务能力，DPO 再优化正确答案相对错误答案的排序。直接 Base-DPO 当然可以，所以保留该消融；预期风险是 Base 的 rejected 质量过低、JSON 合法率低，偏好学习会把容量花在格式和基础标签上，而不是困难坏例。最终以消融指标为准，不把“必须先 SFT”写成先验结论。

**强化学习提升了多少？**

只报告同一 val/test 协议下的绝对增量和相对错误下降，例如 `Delta AUC = AUC_DPO - AUC_SFT`、`badcase reduction = (N_SFT - N_DPO) / N_SFT`。在真实实验完成前，所有结果栏保持空白。

## 8. 交付物

- 可复现的数据转换和泄漏保护。
- LLaMA-Factory 四组训练配置。
- 可产生连续 AUC/mAP 分数的 Qwen3-VL 推理脚本。
- 官方指标兼容评估器和 badcase 报告。
- 实验台账模板，不含虚构结果。
