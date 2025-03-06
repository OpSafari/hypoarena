# 参考文献

本项目借鉴的是公开工作中描述的**机制思路**（生成-辩论-演化循环、锦标赛式排名、
流水线化的研究 agent、树式搜索），schema、算法实现与测试均为本仓库原创，
未复制任何上游代码或大段文本。

## 多 agent 发现循环

- Google AI co-scientist 方向的多 agent 假设生成与锦标赛排名机制：
  <https://arxiv.org/abs/2502.18864>
  借鉴点：generate / debate / evolve 三阶段拆分，以及用成对比较 + Elo 类评分
  为假设排序的思路。本项目自行实现 rubric、评分更新与调度。

## 研究流水线编排

- Agent Laboratory：把研究流程拆成可检查阶段的流水线设计：
  <https://arxiv.org/abs/2402.14207>
  借鉴点：阶段化产物与人在环检查点的组织方式。
- AI Scientist：端到端研究流水线的组件划分与其局限性讨论：
  <https://arxiv.org/abs/2408.06292>
  借鉴点：对“自动化研究”边界的谨慎表述，本项目的诚实性说明与之呼应。

## 搜索式推理

- Tree-of-Thought 式搜索：把候选思路组织成可回溯的树：
  <https://arxiv.org/abs/2305.14251>
  借鉴点：候选生成-评估-展开的控制流；本项目在假设演化算子中使用类似结构。

## 成对比较评分

- Elo  rating 系统与 Bradley–Terry 模型：成对比较结果的极大似然评分方法，
  以及 K 因子调度与平局处理的标准做法（见 ELO 1978 *The rating of chessplayers, past and present*；
  Bradley & Terry 1952, *Rank analysis of incomplete block designs*）。
  本项目自行实现评分更新、平局处理、K 因子衰减与审计轨迹。

## 引用约定

上述链接仅用于说明机制来源。若在学术工作中引用本仓库，请引用仓库本身
（<https://github.com/Netrixo/hypoarena>）而不是把它当作上述工作的实现。
