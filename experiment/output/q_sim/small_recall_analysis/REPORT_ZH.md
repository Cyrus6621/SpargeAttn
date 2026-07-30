# matched fallback 核心指标

## 实验定义

- 危险程度只由原始 `q` 计算：每个 Q block 内的 query token 分别选择 K blocks，并用 `D_adj@50%` 衡量选择分歧。
- 在每个 `document × layer × Q-head` 内，将 `D_adj@50%` 最高的 20% Q blocks 定义为高分歧。
- raw gate 用 `1-selfsim(q)` 排 Q blocks；q-kbar gate 用 `1-selfsim(q-k̄)` 排同一批 Q blocks。
- 两个 gate 在每组 fallback 完全相同数量的 Q blocks。
- `q-k̄` 从不参与 K-block 选择，只用于判断 Q block 是否 fallback。

## 四格定义

- TP：高分歧，并且被 fallback；这是成功救回。
- FP：不是高分歧，但被 fallback；这是浪费的 dense 预算。
- FN：高分歧，但没有 fallback；这是漏掉的危险 block。
- TN：不是高分歧，也没有 fallback。

## 主结果（文档等权）

| nominal fallback | gate | precision | recall | F1 | fallback 中非高分歧比例 |
|---:|---|---:|---:|---:|---:|
| 10% | $1-s(q)$ | 0.391 | 0.214 | 0.275 | 60.9% |
| 10% | $1-s(q-\bar{k})$ | 0.482 | 0.262 | 0.338 | 51.8% |
| 20% | $1-s(q)$ | 0.356 | 0.356 | 0.356 | 64.4% |
| 20% | $1-s(q-\bar{k})$ | 0.428 | 0.428 | 0.428 | 57.2% |

解释：precision=TP/(TP+FP)，回答“fallback 的 blocks 中有多少确实是高分歧”；recall=TP/(TP+FN)，回答“所有高分歧 blocks 中找回了多少”；F1=2×precision×recall/(precision+recall)。这些量先逐文档计算再等权平均，所以表中 F1 不必严格等于把表中宏平均 precision 和 recall 再代入一次公式。

## q-kbar 相对 raw Q

| nominal fallback | Δprecision | Δrecall | ΔF1 |
|---:|---:|---:|---:|
| 10% | +0.091 | +0.048 | +0.063 |
| 20% | +0.072 | +0.072 | +0.072 |

## 为什么 recall 看起来低

总漏检可以严格拆成两部分：

```text
1 - recall = budget-forced miss + ranking miss
budget-forced miss = 1 - 理论最大 recall
ranking miss = 理论最大 recall - 实际 recall
```

| nominal fallback | gate | 理论最大 recall | 预算强制漏检 | 排序造成漏检 | 总漏检 |
|---:|---|---:|---:|---:|---:|
| 10% | $1-s(q)$ | 0.544 | 0.456 | 0.330 | 0.786 |
| 10% | $1-s(q-\bar{k})$ | 0.544 | 0.456 | 0.282 | 0.738 |
| 20% | $1-s(q)$ | 1.000 | 0.000 | 0.644 | 0.644 |
| 20% | $1-s(q-\bar{k})$ | 1.000 | 0.000 | 0.572 | 0.572 |

10% 预算的理论最大 recall 只有约 0.54，因为危险集合约占 20%，fallback 数量只有它的一半左右；这是不可避免的容量限制。但 q-kbar 实际 recall 只有约 0.26，剩余差距仍来自排序不够准。

20% 预算与危险集合大小相同，理论上可以达到 recall=1；q-kbar 实际约为 0.43，说明此时低 recall 完全是单一 self-sim 分数的排序能力有限，而不是预算不够。

## 所有 block 合并后的直观 10% 混淆比例

- raw Q：TP=4.20%，FP=6.58%，FN=16.48%，TN=72.74%。
- q-kbar：TP=5.37%，FP=5.41%，FN=15.31%，TN=73.91%。

所以“约 6% 不是高分歧”必须解释为占全部 Q blocks 的 FP：raw Q 为 6.58%，q-kbar 为 5.41%。若分母改成 fallback 集合，则分别有 61.0% 和 50.2% 不是高分歧。

由于每个较短分组都向上取整选择至少一个 block，nominal 10% 在全体 block 合并后实际为 10.78%；两个 gate 的实际数量严格相同。

## 统计口径

- 完整样本：973 篇；可分析文档：959 篇；总 rows：64,954,368。
- 主表先在每篇文档内平均全部 layer/head，再让每篇文档等权。
- 95% CI 使用 2,000 次 whole-document bootstrap，seed=20260730；详细区间保存在 `core_summary.csv` 和 `qkbar_minus_raw.csv`。
- `pooled_confusion.csv` 只用于把 TP/FP/FN/TN 转成直观的全体 block 比例，不作为文档级推断单位。
