# 可选可训练 ranker（torch extra）

`hypoarena.ranker`（NumPy 层）与 `hypoarena.ranker_torch`（torch 层）一起演示
一个**可训练排序器的机制**：从词面特征预测评审 rubric 分数，训练损失真实下降，
并给出校准曲线。

> **诚实边界**：这个模型学的是我们**植入的合成"质量"信号**，不是科学真伪，
> 也不是任何真实模型的能力。所有数字都来自带植入真值的合成语料，绝非基准成绩。
> 只用 CPU、固定随机种子，不下载任何预训练权重。

## 为什么分两个模块

NumPy 层（数据集、特征、校准、Spearman）不依赖 torch，`import hypoarena.ranker`
在没装 extra 的环境也能成功；只有 `hypoarena.ranker_torch` 需要 torch。所有依赖
torch 的测试都标了 `@pytest.mark.model`，训练类再标 `@pytest.mark.slow`，缺 torch
时用 `pytest.importorskip` 干净跳过。

## 植入的数据集

`synthetic_ranking_dataset(config)` 用标准库 `Random(seed)` 生成陈述：每个 **signal**
token 以 1/2 概率出现，标签是 `planted_quality` —— 出现的 signal token 的**加权占比**
（`signal_weights` 归一化到 `[0,1]`），**noise** token 不携带任何标签信息。因此标签是
特征的确定函数，小模型能真正学到，损失下降是真实的而非数值巧合。相同 seed 逐条一致。

## 特征

`RankerFeaturizer` 复用 `hypoarena.dedup.TfidfVectorizer`（词 1-gram、L2 归一化），
产出 `RankerData(features, scores, vocabulary)`，构造时校验矩阵与标签行数、词表宽度一致。

## 模型与训练

`RubricRanker` 是极小的前馈网络（`input → hidden(16) → 1`，sigmoid 输出，落在
`[0,1]`）。`train_ranker` 用全批量 Adam + MSE，`torch.manual_seed(seed)` 固定初值，
CPU 上**逐 epoch 确定**；返回完整 `losses` 历史，`loss_decreased()` 供测试断言。
`fit_ranker` 训练→预测→报告 Spearman，是**样本内**拟合，只为证明机制能学到植入信号。

## 指标

- `spearman_correlation`：秩相关，ties 取平均秩；输入为常量时返回 `0.0` 而非 `nan`
  （诚实地表示"无单调一致"，不污染均值）。
- `calibration_curve`：按预测分等宽分箱，比较每箱的 `mean_predicted` 与 `mean_actual`；
  空箱也保留（count 0），曲线恒有 `bins` 项，不会靠丢箱掩盖缺口。

## 实测结果（合成数据，非基准）

在 `n=128, seed=270106` 的合成数据集上（由 `test_ranker_recovery.py` /
`test_ranker_training.py` 实际执行）：训练损失降到初值的一半以下；预测与植入质量的
Spearman ρ > 0.8；预测分最高五分位的**植入**质量均值比最低五分位高 0.2 以上。
这些数字只说明"机制在这份合成数据上生效"，不代表任何真实科学判断能力。
