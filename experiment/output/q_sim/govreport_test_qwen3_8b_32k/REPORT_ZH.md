# Q self-sim 与块内 K-block 选择分歧：GovReport 全量实验

只看 `q-k̄` 是否更适合判断 fall-to-dense 的证据，请见
`FALLBACK_EVIDENCE_ZH.md`。

## 结论

### 1. 低 `selfsim(q)` 的 Q block，块内 token 选到的 K blocks 确实更不一致

答案是**是**，而且在 10%、25%、50% 三种 top-block 比例下都成立。
以 risk=`1-selfsim(q)` 作为自变量，在每个
`document × layer × Q-head` 内控制 Q-block 位置后：

| 目标 | partial Spearman ρ [95% CI] | 最高风险四分位减最低风险四分位 |
|---|---:|---:|
| `D_adj@10%` | 0.357 [0.353, 0.361] | +0.082 [0.081, 0.084] |
| `D_adj@25%` | 0.358 [0.354, 0.362] | +0.069 [0.068, 0.070] |
| `D_adj@50%` | 0.341 [0.337, 0.345] | +0.061 [0.060, 0.062] |
| pooled miss `@10%` | 0.324 [0.320, 0.328] | +0.073 [0.072, 0.075] |
| pooled miss `@25%` | 0.330 [0.326, 0.333] | +0.058 [0.057, 0.060] |
| pooled miss `@50%` | 0.319 [0.315, 0.323] | +0.051 [0.050, 0.052] |
| preference JSD | 0.515 [0.511, 0.519] | +0.019 [0.019, 0.020] |

七个目标的 95% whole-document bootstrap CI 都不跨 0。因此这不是只由
长上下文位置带来的假相关：低 self-sim Q block 内的 query token 的确更倾向于
不同的 K blocks，而且用 pooled Q 产生一个共享稀疏 mask 时也更容易漏掉
token 自己想选的 blocks。

该现象也不是少数 layer/head 驱动：`D_adj@10/25/50%` 的
position-controlled ρ 分别在 1135/1152、1144/1152、1143/1152 个
layer×head 单元为正，且 36 层各自对 head 平均后全部为正。

### 2. 若用途是固定预算 dense fallback，`selfsim(q-k̄)` 更好

这里的“更好”定义为：在完全相同的 5%/10%/20% fallback Q-block 预算下，
找回真实 `D_adj` 或 pooled-mask miss 最差的 20% Q blocks。不同特征不共用
绝对数值阈值，只比较排序。

主指标 `D_adj@50%`：

| gate risk | AUROC | AP | recall@10 | lift@10 | recall@20 | lift@20 |
|---|---:|---:|---:|---:|---:|---:|
| `1-selfsim(q)` | 0.652 | 0.425 | 0.214 | 1.821 | 0.356 | 1.656 |
| `1-selfsim(q-k̄)` | **0.715** | **0.501** | **0.262** | **2.258** | **0.428** | **2.000** |
| `q-k̄` 减 `q` | **+0.063** | **+0.076** | **+0.048** | **+0.438** | **+0.072** | **+0.344** |

其中 recall@10 的差值 95% CI 为 `[+0.047,+0.050]`，recall@20 为
`[+0.069,+0.074]`。在 `D_adj@10/25/50%` 三个目标上，`q-k̄` 相对 raw Q：

| 目标 | ΔAUROC | ΔAP | Δrecall@10 | Δrecall@20 |
|---|---:|---:|---:|---:|
| `D_adj@10%` | +0.060 | +0.069 | +0.043 | +0.065 |
| `D_adj@25%` | +0.070 | +0.084 | +0.052 | +0.079 |
| `D_adj@50%` | +0.063 | +0.076 | +0.048 | +0.072 |

pooled miss 的 10%、25%、50% 三个目标也全部得到正向提升，且所有
AUROC、AP、recall@10、recall@20 的 paired document-bootstrap CI 都排除 0。
所以，如果 gate 的任务是决定“哪些 Q blocks 要退回 dense/更高预算”，
建议使用 `1-selfsim(q-k̄)`，而不是直接使用 `1-selfsim(q)`。

### 3. 但 `q-k̄` 不是对所有异质性定义都无条件更好

- 控制 Q-block 位置后，`D_adj@50%` 的 partial ρ 是 raw Q 的 0.341，
  `q-k̄` 的 0.334；差值为 `-0.007 [-0.009,-0.005]`。这说明 `q-k̄`
  的 operational gain 部分来自更有用的跨位置全局排序，而不是每个位置内
  的单调关系都更强。
- 对连续 softmax preference 的 JSD，raw Q 的 operational ranking 更好：
  AUROC `0.753 vs 0.736`，AP `0.547 vs 0.534`，recall@20
  `0.473 vs 0.460`。但 JSD 的 position-controlled partial ρ 反而由
  0.515 提升到 0.537。

因此建议是目标特定的：

- **离散 top-K block 选择是否分裂、共享 mask 是否会漏、是否触发 dense
  fallback：用 `q-k̄`。**
- **若目标是连续 preference-distribution JSD：保留 raw Q，或对目标单独
  校准，不应直接替换。**

## 实验覆盖

- 模型：`Qwen3-8B`，全部 36 层、32 个 Q heads、8 个 KV heads，
  head dim 128；每 4 个 Q heads 映射到一个 KV head。
- 数据：本地 GovReport test **全部 973 篇**。
- 处理 token：9,266,582 / 原始 9,290,399；最大长度 32,768。
- 4 篇超过 32K 被截断；959 篇产生满足预注册条件的观测；14 篇太短，
  没有 `M≥32` 个共同可见 K blocks，保留完整 metadata 但不进入关联估计。
- 观测总数：64,954,368 条 `document × layer × Q-head × Q-block` rows。
- 所有非空文档均覆盖全部 layer/head；973 个 Parquet 和 973 个 metadata
  文件齐全，无失败样本。
- 统计单位是 document：先在每个 document/layer/head 内计算，再按文档
  等权宏平均；置信区间使用 2,000 次 paired whole-document bootstrap，
  没有把 6,495 万个 block row 当作独立样本。

本地 RULER plain 有 39,000 条，而且其中 64K/128K 设置超出该 checkpoint
未启用 YaRN 时的原生位置范围。本次选择可完整遍历且与长文摘要相符的
GovReport test；结论不外推到 RULER 或 32K 以上上下文。

## 指标与因果边界处理

Q128/K64 与当前 A100/A800 SpargeAttn block 设置一致。Q、K 都从模型真实
forward 中 QK-norm 和 RoPE 之后、attention 之前截取。

对一个 Q block，只允许选择在该 block 第一个 query token 之前已经完整结束的
K64 blocks。这样 128 个 query token 的候选集合完全相同，不会把 causal
boundary 导致的可见 K 数量差异误认为 query 异质性。

self-sim 定义为：

```text
selfsim(X) = || mean_t normalize(x_t) ||²
```

`k̄` 是对应 GQA KV head 在保留的完整 prefill 序列上的均值。selector target
始终由原始 `q` 计算：

```text
score(t,j) = q_t · mean(K_j-k̄) / sqrt(128)
```

减去同一个 `k̄` 不改变 K-block 排名；`q-k̄` 只改变 gate 特征，不改变
用于评价的 selector，因而没有用 alternate feature 重定义答案。

主目标是 chance-adjusted token-pair top-set disagreement：

```text
overlap = Σ_j C(c_j,2) / (C(128,2) · k)
D_adj   = (1-overlap) / (1-k/M)
```

`D_adj=0` 表示块内所有 query token 选择完全相同；`D_adj=1` 约等于随机
top-k 的分歧程度。另报告 pooled-Q mask miss 和 preference JSD，避免只依赖
单一离散指标。

## 对照实验

以 `D_adj@50%` 为例：

| gate risk | partial ρ | AUROC | AP | recall@10 | recall@20 |
|---|---:|---:|---:|---:|---:|
| `1-selfsim(q)` | 0.341 | 0.652 | 0.425 | 0.214 | 0.356 |
| `1-selfsim(q-k̄)` | 0.334 | **0.715** | **0.501** | **0.262** | **0.428** |
| `1-selfsim(q-q̄)` | 0.271 | 0.655 | 0.440 | 0.224 | 0.362 |
| `1-selfsim(q-k̄_wrong)` | 0.308 | 0.627 | 0.415 | 0.205 | 0.340 |

`q-q̄` 没有复现提升，减去下一个错误 KV head 的 `k̄_wrong` 反而变差；
因此 operational gain 不是任意平移或中心化都能得到，而与正确配对的
KV-head `k̄` 有关。

平均提升并不代表每个 head 都提升：例如 `D_adj@50%` 的 AUROC 在
806/1152 个 layer×head 单元中由 `q-k̄` 胜出。实际部署可以先用
document-macro 结论作为默认，再在独立校准集上考虑 per-layer/head gate。

## 绝对量级

以下先在每篇文档内算均值/分位数，再对文档等权平均：

| 变量 | mean | p10 | p50 | p90 |
|---|---:|---:|---:|---:|
| raw Q self-sim | 0.6276 | 0.4653 | 0.6360 | 0.7829 |
| `q-k̄` self-sim | 0.7610 | 0.5650 | 0.7790 | 0.9283 |
| `D_adj@10%` | 0.5366 | 0.3303 | 0.5304 | 0.7570 |
| `D_adj@25%` | 0.5136 | 0.3128 | 0.5119 | 0.7206 |
| `D_adj@50%` | 0.5139 | 0.3004 | 0.5219 | 0.7142 |
| pooled miss `@50%` | 0.3824 | 0.2159 | 0.3834 | 0.5464 |
| JSD | 0.0692 | 0.0266 | 0.0586 | 0.1267 |

raw Q 与 `q-k̄` 的数值分布明显不同，所以部署时不能共享一个未经重新校准
的绝对阈值。

## 产物

- `REPORT.md`：英文完整报告。
- `analysis/validation.json`：完整性和结构检查。
- `analysis/macro_summary.csv`：全部 predictor × target × metric 结果和 CI。
- `analysis/qkbar_minus_raw_bootstrap.csv`：`q-k̄` 减 raw Q 的配对差值。
- `analysis/document_metrics.parquet`：按文档聚合的统计单位。
- `analysis/layer_head_summary.parquet`：layer/head 诊断。
- `plots/fallback_recall.png`：固定 fallback 预算对比。
- `plots/partial_spearman_heatmap.png`：控制位置后的关联矩阵。
- `plots/qkbar_minus_raw_bootstrap.png`：配对差值及 95% CI。
- `samples/`：逐文档原始 block-level 统计，可继续做不同阈值或新指标分析。

## 范围限制

本实验评价的是 SpargeAttn-style K-block centroid preference，不是 dense
token attention mass oracle，也不是端到端生成误差或速度 benchmark。
`k̄` 是当前 prefill smoothing 路径使用的整段 retained-sequence mean；
自回归 decoding 中的 causal running mean 需要另做实验。当前结果支持
matched-budget gate 排序，不提供可跨数据集直接使用的固定数值阈值。
