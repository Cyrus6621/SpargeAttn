# Q self-similarity and within-block key-selection disagreement

## Main result

Across all seven targets, lower raw-Q self-similarity is positively associated with worse within-block key behavior even after controlling position (all document-bootstrap CIs exclude 0).

For the operational gate question, q-kbar is better on every discrete top-set
disagreement and pooled-mask-miss target: at the same 10%/20% dense-fallback
budget, all 12 recall contrasts are positive and their paired whole-document
bootstrap intervals exclude 0. For `D_adj@50%`, q-kbar improves AUROC by 0.063,
AP by 0.076, recall@10 by 0.048, and recall@20 by 0.072.

This is not a universal replacement result. For position-controlled partial
Spearman, the comparison is target-dependent: 1 target favors q-kbar, 3 favor
raw, and 3 overlap zero. Raw Q also ranks graded JSD risk better in the
operational AUROC/AP/recall view.

These statements are generated from the estimates below; no expected winner is hard-coded.

## Coverage and validity

- Input: `/home/dangyunkai/yunkai/VLM/VIG-Group/haoyi/ICLR27/sparse_attn/SpargeAttn/experiment/output/q_sim/govreport_test_qwen3_8b_32k`
- Collector sample files: 973 / 973 expected.
- Documents with eligible Q blocks: 959.
- Pre-registered short documents with zero eligible Q blocks: 14 (indices: [153, 401, 415, 553, 635, 695, 711, 715, 717, 731, 797, 836, 961, 971]).
- Nonempty documents with restricted layer/head coverage: 0.
- Total collected Q-block rows: 64,954,368.
- Documents truncated at the configured context cap: 4 (indices: [189, 440, 694, 876]).
- Tokens processed: 9,266,582 / 9,290,399 before truncation.
- Model: `/home/dangyunkai/yunkai/VLM/VIG-Group/model/Qwen3-8B`; metadata says 36 layers and 32 Q heads.
- Validation status: **complete**.

Zero-row samples are valid only when their token length implies no full Q block with the configured minimum common-key prefix. They count toward file completeness but not toward association estimates.

Every analyzed Q block has at least `M >= 32` fully visible key blocks. The candidate set is the common causal prefix visible to every token in that Q block, so causal-boundary availability is held fixed within the group. Partial Spearman additionally controls ranked Q-block position.

## Core estimates

Each cell is the document-macro estimate with a 95% whole-document bootstrap interval.

| Target | raw partial Spearman | raw high-risk minus low-risk target quartile | q-kbar minus raw partial Spearman |
|---|---:|---:|---:|
| $D_{adj}@10\%$ | 0.357 [0.353, 0.361] | 0.082 [0.081, 0.084] | 0.001 [-0.001, 0.003] |
| $D_{adj}@25\%$ | 0.358 [0.354, 0.362] | 0.069 [0.068, 0.070] | -0.001 [-0.003, 0.000] |
| $D_{adj}@50\%$ | 0.341 [0.337, 0.345] | 0.061 [0.060, 0.062] | -0.007 [-0.009, -0.005] |
| Pooled miss$_{adj}@10\%$ | 0.324 [0.320, 0.328] | 0.073 [0.072, 0.075] | -0.001 [-0.002, 0.001] |
| Pooled miss$_{adj}@25\%$ | 0.330 [0.326, 0.333] | 0.058 [0.057, 0.060] | -0.003 [-0.004, -0.001] |
| Pooled miss$_{adj}@50\%$ | 0.319 [0.315, 0.323] | 0.051 [0.050, 0.052] | -0.008 [-0.010, -0.006] |
| JSD | 0.515 [0.511, 0.519] | 0.019 [0.019, 0.020] | 0.022 [0.020, 0.023] |

## Primary operational view: adjusted disagreement at 50%

Document-macro estimate [95% whole-document bootstrap CI]. Higher AUROC/AP/recall/lift means the risk score finds more of the worst `D_adj_r50` Q blocks at the same fallback budget.

| Predictor | partial rho | AUROC worst20 | AP worst20 | recall@10 | lift@10 | recall@20 | lift@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| $1-s(q)$ | 0.341 [0.337, 0.345] | 0.652 [0.650, 0.654] | 0.425 [0.420, 0.430] | 0.214 [0.210, 0.219] | 1.821 [1.804, 1.838] | 0.356 [0.353, 0.360] | 1.656 [1.643, 1.669] |
| $1-s(q-\bar{k})$ | 0.334 [0.330, 0.338] | 0.715 [0.712, 0.717] | 0.501 [0.498, 0.505] | 0.262 [0.259, 0.267] | 2.258 [2.237, 2.279] | 0.428 [0.425, 0.432] | 2.000 [1.984, 2.016] |

## Absolute scale of self-similarity and targets

These are descriptive document-macro summaries: each mean or quantile is computed within a document first, then averaged across documents. They show absolute scale without treating Q blocks as independent observations.

| Variable | document-macro mean | p10 | p50 | p90 |
|---|---:|---:|---:|---:|
| raw Q self-sim | 0.6276 | 0.4653 | 0.6360 | 0.7829 |
| q-kbar self-sim | 0.7610 | 0.5650 | 0.7790 | 0.9283 |
| $D_{adj}@10\%$ | 0.5366 | 0.3303 | 0.5304 | 0.7570 |
| $D_{adj}@25\%$ | 0.5136 | 0.3128 | 0.5119 | 0.7206 |
| $D_{adj}@50\%$ | 0.5139 | 0.3004 | 0.5219 | 0.7142 |
| Pooled miss$_{adj}@10\%$ | 0.4101 | 0.2360 | 0.3976 | 0.6057 |
| Pooled miss$_{adj}@25\%$ | 0.3855 | 0.2246 | 0.3787 | 0.5585 |
| Pooled miss$_{adj}@50\%$ | 0.3824 | 0.2159 | 0.3834 | 0.5464 |
| JSD | 0.0692 | 0.0266 | 0.0586 | 0.1267 |

## Metrics and estimand

- Predictor risk is `1 - self_similarity`; larger values mean lower within-Q-block self-similarity.
- Spearman is computed across Q blocks inside each document × layer × Q-head group.
- Partial Spearman is the correlation of predictor and target ranks after linearly residualizing ranked Q-block position.
- Quartile delta is mean target in the highest-risk predictor quartile minus mean target in the lowest-risk quartile.
- AUROC and average precision classify the worst 20% target Q blocks within the same group.
- Matched fallback selects exactly the highest-risk 5%, 10%, or 20% Q blocks (ceil-rounded) for every predictor. Recall is the fraction of worst-20% target blocks caught; lift is recall divided by the realized selected fraction.
- Group metrics are averaged over layer/head groups inside each document. Dataset estimates then average documents equally.
- Confidence intervals use 2,000 paired whole-document bootstrap replicates (seed 20260729). No Q block is treated as an independent replicate.

The selector targets were always computed from original Q. `q-kbar`, `q-qbar`, and wrong-`kbar` alter only the gating predictor, preserving a clean comparison.

## Scope and limitations

- The target is token-wise preference over SpargeAttn-style K-block centroids. It is not a dense token-attention-mass oracle, an end-to-end output-error measurement, or a speed benchmark.
- `kbar` is the full retained prefill-sequence mean used by the current SpargeAttn smoothing path. Autoregressive decoding with a causal running mean is a separate setting.
- Results cover the complete local GovReport test split under a 32K cap. RULER task transfer and contexts beyond the checkpoint's native range are not inferred from this run.
- Associations and matched-budget retrieval metrics establish gate usefulness, not a universal fixed numeric threshold. Any deployment threshold should be calibrated on separate documents.
- The q-kbar operational gain is a document-macro average, not a claim that every layer/head improves; per-layer/head deployment choices need separate calibration.

## Output files

- `analysis/validation.json`: completeness and structural checks.
- `analysis/document_metrics.parquet`: one row per observed document × predictor × target after layer/head averaging.
- `analysis/document_distribution.parquet` and `distribution_macro.csv`: per-document and document-macro absolute means/quantiles.
- `analysis/macro_summary.csv`: all estimates and document bootstrap intervals.
- `analysis/qkbar_minus_raw_bootstrap.csv`: paired q-kbar-minus-raw contrasts.
- `analysis/layer_head_summary.parquet`: document-macro layer/head diagnostics.
- `plots/`: association, contrast, fallback, and layer figures.
