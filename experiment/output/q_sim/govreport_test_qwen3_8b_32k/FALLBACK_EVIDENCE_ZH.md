# `selfsim(q-k̄)` 用于判断 fall-to-dense 的证据

## 被数据支持的准确结论

在本实验测试的四种 self-sim gate 中，如果 fallback 的目标是发现：

1. Q block 内不同 query token 的 top-K key-block 选择发生分裂；或
2. pooled-Q 共享 mask 会漏掉 token 自己选择的 key blocks，

那么 **`1-selfsim(q-k̄)` 是最好的 fall-to-dense 风险分数**。

这里不声称 `q-k̄` 更适合直接选择 K blocks。所有评价 target 始终由原始
`q` 选择 K blocks；`q-k̄` 只用于给 Q block 排 fallback 风险。

## 公平比较方式

- 四个 gate：
  - `1-selfsim(q)`
  - `1-selfsim(q-k̄)`
  - `1-selfsim(q-q̄)`
  - `1-selfsim(q-k̄_wrong)`
- 对每种 gate 都只 fallback 风险最高的相同 5%、10% 或 20% Q blocks。
- “需要 fallback”的真值是对应 `D_adj` 或 pooled miss 最高的 20% Q blocks。
- 比较 AUROC、AP、recall@10 和 recall@20；越高越好。
- 先在每个 `document × layer × Q-head` 内比较，再按文档等权平均。
- 差值置信区间使用 2,000 次 paired whole-document bootstrap。

因此结果不是由某个 gate 使用了更多 dense 预算、某些长文档权重更高或
`q-k̄` 自己重新定义评价 target 造成的。

## 汇总证据

共有：

```text
6 个离散 target
× 4 个 operational metric
= 24 个 target × metric 比较
```

六个 target 是：

```text
D_adj@10%、D_adj@25%、D_adj@50%
pooled miss@10%、pooled miss@25%、pooled miss@50%
```

结果：

- **24/24 个比较中，`q-k̄` 的点估计都是四个 gate 中最高。**
- **24/24 个 `q-k̄ - raw q` 差值的 95% CI 下界都大于 0。**
- 其中 12/12 个 matched-budget recall 差值也全部显著大于 0。

这说明“`q-k̄` 更适合决定 fall-to-dense”不是依赖某一个 top-K 比例或某一个
统计指标得出的。

## 主指标：`D_adj@50%`

| gate | AUROC | AP | recall@10 | recall@20 |
|---|---:|---:|---:|---:|
| `1-selfsim(q-k̄)` | **0.715** | **0.501** | **0.262** | **0.428** |
| `1-selfsim(q-q̄)` | 0.655 | 0.440 | 0.224 | 0.362 |
| `1-selfsim(q)` | 0.652 | 0.425 | 0.214 | 0.356 |
| `1-selfsim(q-k̄_wrong)` | 0.627 | 0.415 | 0.205 | 0.340 |

相对 raw Q，`q-k̄` 的提升为：

| metric | `q-k̄ - raw q` [95% CI] |
|---|---:|
| AUROC | +0.0628 [+0.0609, +0.0647] |
| AP | +0.0764 [+0.0740, +0.0789] |
| recall@10 | +0.0482 [+0.0465, +0.0499] |
| recall@20 | +0.0718 [+0.0694, +0.0741] |

也就是说，只 fallback 10% Q blocks 时，`q-k̄` 比 raw Q 多找回约 4.8 个
百分点的真实高分歧 blocks；fallback 20% 时多找回约 7.2 个百分点。

## 所有离散 target 的 matched-budget recall

下表都是 `q-k̄ - raw q`：

| target | Δrecall@10 [95% CI] | Δrecall@20 [95% CI] |
|---|---:|---:|
| `D_adj@10%` | +0.0434 [+0.0417,+0.0451] | +0.0653 [+0.0629,+0.0679] |
| `D_adj@25%` | +0.0523 [+0.0506,+0.0541] | +0.0793 [+0.0768,+0.0817] |
| `D_adj@50%` | +0.0482 [+0.0465,+0.0499] | +0.0718 [+0.0694,+0.0741] |
| pooled miss `@10%` | +0.0383 [+0.0367,+0.0400] | +0.0589 [+0.0566,+0.0615] |
| pooled miss `@25%` | +0.0507 [+0.0490,+0.0524] | +0.0775 [+0.0751,+0.0800] |
| pooled miss `@50%` | +0.0475 [+0.0459,+0.0491] | +0.0715 [+0.0691,+0.0737] |

所有置信区间都完全位于 0 以上。

## 为什么对照项重要

- `q-q̄` 没有达到 `q-k̄` 的效果，说明不是“任意中心化”都能提升。
- 减去错误 KV head 的 `k̄_wrong` 反而比 raw Q 更差。
- 因此提升与正确 GQA KV head 的 `k̄` 有关，而不是简单平移带来的假象。

## 证据边界

该证据支持：

> 原始 `q` 负责选择 K blocks，`selfsim(q-k̄)` 负责判断该 Q block 是否应
> fall back to dense。

该证据不支持：

- `q-k̄` 直接选择 K blocks 比原始 `q` 更好；本实验没有这样测试。
- `q-k̄` 对所有异质性指标都最好；对于连续 preference JSD，raw Q 的
  operational ranking 更好。
- 一个无需校准即可跨数据集使用的固定 self-sim 阈值。

## 原始证据表

- `analysis/macro_summary.csv`：四个 gate 的全部绝对指标和置信区间。
- `analysis/qkbar_minus_raw_bootstrap.csv`：全部 `q-k̄ - raw q` 配对差值。
- `analysis/document_metrics.parquet`：document-level 统计单位。
- `plots/fallback_recall.png`：相同 fallback 预算下的可视化比较。
