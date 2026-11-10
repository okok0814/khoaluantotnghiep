# Baseline and SOTA comparison audit

**Weekly report, 10 November 2026:** [Vietnamese baseline and published-method comparison report](../logs/paper_comparisons/weekly_report.md), with verified measured tables, published references, limitations, and code/evidence links.

**Evidence update, 5 November 2026:** the new Colab archive contains completed full-validation SPRC ViT-L (49.88/71.36), CLIP4CIR RN50x4 (40.87/64.37), matched-file fixed baselines, three-seed learned controls, and UniFashion initialization. Checksums and complete per-query predictions were verified. These close several execution gaps recorded below. The [weekly report](../logs/paper_comparisons/weekly_report.md) presents the non-gated results, with [verified tables](../papers/fashion_search/evidence/verified_results.csv) and explicit published-protocol/resource qualifications. It does not claim SOTA superiority; the integrated teammate model remains outside this report's scope.

**Implementation update, 4 November 2026:** the historical standalone gate has now been re-evaluated under the shared protocol (38.76/60.85 macro R@10/R@50), and the six measured variants have been consolidated with paired query intervals. See the [comparison report](../logs/paper_comparisons/report/comparison_report.md) and [T4 runbook](Conference_Comparisons_Runbook.md). The sections below preserve the audit's original findings; external-checkpoint evaluations and new controls remain pending GPU execution, and the integrated UniFashion gate checkpoint remains unavailable.

Reviewed 4 November 2026, local commit `662af9e`. This audit covers the source/output inventory of all 17 notebooks, retrieval and training scripts, configs, saved evaluation artifacts, current and historical reports, the earlier extracts of ten supplied weekly PDFs, and the public GitHub README. It recalculates metrics from saved predictions; it does not run new model inference or training.

**Finding: measured baseline comparisons exist. A broad, controlled comparison with independent published CIR methods is still missing.** The advisor's concern is justified about experimental breadth and comparability. It would be inaccurate to describe the project as having no baseline experiments at all.

| Question | Finding |
|---|---|
| Have simple baselines been evaluated? | Yes: FashionCLIP image-only, text-only and additive, with full per-query validation results. |
| Has a learned fusion method been compared with additive? | Yes: a separate FashionCLIP gate notebook has results; it needs protocol reconciliation before joining the main table. |
| Has fine-tuned UniFashion been compared with baselines? | Yes, at 29,500, 60,000 and 90,000 updates. |
| Have published numbers been discussed? | Yes, principally UniFashion's 58.93/76.93 reference in the tuning reports. |
| Have several external CIR methods been run under the shared protocol? | No completed result artifacts found for CLIP4CIR/Combiner, SPRC, CASE, FAME-ViL, QuRe or similar independent methods. |
| Has the integrated UniFashion attention-fusion improvement been established? | No full-validation Recall artifact found for that model variant. |
| Can the current evidence establish SOTA or a new fusion method's advantage? | No; additional comparison and ablation evidence is needed. |

## 1. Comparisons that are already supported

All figures below are **macro percentages**, averaging dress, shirt and toptee equally. The common protocol contains 6,016 validation queries, full category galleries and reference exclusion. These rows were recalculated from their per-query CSVs, including rank/hit consistency and query coverage.

| Evaluated system | R@10 | R@50 | Role in the paper |
|---|---:|---:|---|
| FashionCLIP image-only | 7.61 | 16.92 | Single-modality baseline |
| FashionCLIP text-only | 27.78 | 47.10 | Single-modality baseline; stronger than fixed additive |
| FashionCLIP additive | 23.62 | 41.54 | Fixed image/text composition baseline |
| UniFashion step29,500, with gallery-caption score | 57.08 | 75.43 | Intermediate checkpoint |
| UniFashion step60,000, with gallery-caption score | 62.57 | 79.18 | Intermediate checkpoint |
| UniFashion step90,000, without gallery-caption score | 45.14 | 68.53 | Caption-scoring ablation of the final model |
| UniFashion step90,000, with gallery-caption score | 64.79 | 80.64 | Final original model, selected over step60,000 |
| Continuation trial LR 1e-5 / batch 4, with caption score | 61.88 | 78.88 | Negative tuning result relative to its step60,000 parent |

Full category, macro and micro results for **11 evaluated variants** are consolidated in [verified_project_results.csv](paper_comparison_audit/verified_project_results.csv). [verification.json](paper_comparison_audit/verification.json) records the scope and input hashes. The older step29,500 metadata predates the `evaluation_scope` field; its protocol, coverage, ranks, step and runtime identity were checked without inventing that missing metadata.

The five core rows for the main results table should be image-only, text-only, additive, final UniFashion without gallery-caption scoring, and final UniFashion with it. Move checkpoint progression and LR/batch trials to separate ablations. Multiple checkpoints of one model do not substitute for independent competing methods.

**Presentation gap:** the latest completion report and plot include additive but omit the already measured image-only and text-only baselines. Text-only exceeds additive by 4.16/5.56 percentage points. Including it avoids making the reference baseline look weaker than the strongest simple baseline available. The final caption-assisted system exceeds text-only by 37.01/33.54 points; the final no-caption-score variant exceeds it by 17.36/21.43 points. These are system comparisons, not isolated estimates of the benefit of fine-tuning.

Evidence and executable definitions:

- [Shared-protocol baseline generation](../scripts/prepare_fashioniq_evaluation.py), [protocol and ranking implementation](../src/fashioniq_evaluation.py), and [three saved baseline results](../logs/week41_fashioniq_evaluation/baselines/).
- [Final checkpoint evaluation](../scripts/evaluate_unifashion_fashioniq.py) and [final evaluation results](../logs/week43_44_unifashion_completion/evidence_completed_20261102/final_results/).
- [Current report](../logs/week43_44_unifashion_completion/weekly_report.md) and [current comparison CSV](../logs/week43_44_unifashion_completion/model_comparison.csv).
- Week41 PDF already contains a direct baseline table. Week43 PDF section 1.4 already discusses the published UniFashion reference, while explicitly qualifying comparability.

## 2. Other experiments must be identified correctly

**Standalone FashionCLIP gate.** [40_16_AttentionFusion_FashionIQ.ipynb](../notebooks/40_16_AttentionFusion_FashionIQ.ipynb) and [its saved CSV](../data/processed/week40_attention_fusion/attention_vs_additive_recall.csv) report macro **38.75/60.85**, versus **23.64/41.54** additive within that notebook. It trains a gate on cached FashionCLIP embeddings for 15 epochs, batch 256, LR 1e-3. These are real saved aggregate results, but no common-protocol per-query gate export was found. Its evaluator can filter missing gallery entries/targets and uses `torch.topk`; the common evaluator rejects missing coverage and uses a stable tie rule. The presence of filtering code does not prove samples were dropped in this run. Re-evaluate the saved checkpoint with full coverage checks before merging its row with the verified table.

**Integrated UniFashion gate.** [41_15_Attention_Fusion_UniFashion_GPU.ipynb](../notebooks/41_15_Attention_Fusion_UniFashion_GPU.ipynb) records a separate completed 90,000-update run, but its output says `retrieval_evaluated: false`; the last evaluation cell only checks `--help`. Training loss is insufficient to compare its retrieval quality. The user's last update also said the intended teammate run was not ready. This notebook does not supply its missing Recall result.

The gate in [attention_fusion.py](../src/attention_fusion.py) is an MLP-generated elementwise gate plus residual projection. The name “attention-fusion” does not by itself establish architectural novelty. Learned gating and residual composition have precedents, including the [official CLIP4CIR Combiner](https://github.com/ABaldrati/CLIP4Cir), which makes it a relevant methodological control.

**Older additive/concat results.** The Week38 PDF reports roughly 32%/54% additive using [the older evaluator](../scripts/evaluate_baseline_fashioniq.py), whose gallery consists only of unique labeled targets and does not explicitly exclude the reference. Those numbers cannot be inserted into the full-gallery table. The separate [Week38 notebook](../notebooks/38_18_FashionIQ_RecallK.ipynb) already uses full galleries and reference exclusion, but reports 23.67/41.54: it defines rank as `1 + count(score > target_score)`, giving tied targets the best tied rank. The standalone gate notebook reports 23.64/41.54, and the shared evaluator reports 23.62/41.54. Use the shared evaluator's results and document the ranking difference; do not treat these small differences as measured model gains without paired verification.

The concat implementation compares `[image; text]` against `[target_image; zeros]`. For a fixed query, its cosine ranking therefore depends only on the image dot product. It is a useful diagnostic, but it does not add an independent, effective text/image fusion baseline. Likewise, extracting ResNet, CLIP or Sentence-BERT embeddings is not evidence of a complete CIR evaluation of each backbone.

## 3. What “comparison with SOTA” currently means here

The architecture survey discusses FashionVLP, FashionViL, FashionSAP, FAME-ViL and UniFashion. That establishes literature coverage, not measured comparative performance. The existing tuning report compares one published UniFashion number with project scores. It does not constitute a broad benchmark of independent methods.

The following are **published reference values checked during this audit**, not additional project experiments. They belong in a separate literature table until the relevant conditions and model variants have been reconciled.

| Published model / variant | R@10 | R@50 | Exact source |
|---|---:|---:|---|
| UniFashion, with captions | 58.93 | 76.93 | [EMNLP 2024, Table 4](https://aclanthology.org/2024.emnlp-main.89.pdf) |
| UniFashion, without caption score | 55.27 | 74.80 | [EMNLP 2024, Table 4](https://aclanthology.org/2024.emnlp-main.89.pdf) |
| SPRC, original full model | 54.92 | 74.97 | [ICLR 2024, Table 1](https://proceedings.iclr.cc/paper_files/paper/2024/file/8204f54f6cd8f0db1765a274e94ceaac-Paper-Conference.pdf) |
| QuRe, BLIP-2 / ViT-L | 52.60 | 73.48 | [ICML 2025 paper, Tables 2 and 7](https://arxiv.org/html/2507.12416v1) |

Machine-readable sources: [published_reference_results.csv](paper_comparison_audit/published_reference_results.csv). This is a small contextual sample, **not an exhaustive November 2026 SOTA leaderboard**. Later work exists, including [DQE-CIR, March 2026 preprint](https://arxiv.org/abs/2603.04037); refresh related work and verify the exact evaluation regime when finalizing the manuscript. Avoid choosing only older or lower-scoring methods. Published tables can use different variants: QuRe's Table 2 lists an SPRC result different from SPRC's original Table 1, so model names alone are insufficient to select numbers.

**The caption discrepancy is scientifically important.** Project step90,000 is +5.86/+3.71 points relative to the paper's caption-assisted row, but −10.13/−6.27 relative to its no-caption-score row. Within the project, adding gallery-caption scoring changes the final checkpoint by +19.64/+12.11 points. This warrants checking caption generation, provenance, text and image processing before any superiority claim. The paper describes evaluation captions generated by trained UniFashion; this project uses released `next_llava.val.*.caption.json` resources and does not demonstrate that caption-generation pipeline. That difference requires investigation; it is not proof of leakage or invalid metrics. [UniFashion, Table 4 and Appendix C](https://aclanthology.org/2024.emnlp-main.89.pdf).

## 4. Remaining comparison gaps

| Gap | Why it matters | Concrete remedy |
|---|---|---|
| No independent trained CIR competitor under the shared protocol | Beating fixed FashionCLIP composition cannot establish superiority to established CIR methods | Evaluate released CLIP4CIR/Combiner and SPRC checkpoints; add an appropriate recent method when feasible |
| No UniFashion before/after fine-tuning Recall control | Architecture/pretraining and fine-tuning are confounded in the FashionCLIP comparison | Evaluate a clearly identified pre-fine-tuning retrieval initialization; disclose any retrieval heads initialized randomly |
| No completed integrated-fusion Recall comparison | Cannot attribute improvement to the new module | Compare original UniFashion against UniFashion + gate using matched initialization, data, budget, selection rule and scoring |
| Learned gate mainly compared with fixed additive | Extra learned capacity and task training may explain its gain | Add weighted additive and a parameter/budget-matched MLP or Combiner control |
| Captions provide extra candidate-side information | A large gain may reflect resources/scoring rather than fusion quality | Report with/without gallery-caption score, document caption generation, and compare systems with their resources clearly labeled |
| Only one original training seed and no uncertainty estimates | No evidence of training stability or statistical reliability of small gains | Add paired query uncertainty analysis; multiple training seeds when feasible, distinguishing the two sources of variation |
| Cached baseline pixels not fully matched to UniFashion image files | Same query/gallery IDs do not establish identical image inputs | Re-extract baseline features from the same audited image files, using each backbone's documented transform |

The original training uses batch 2 / FP32; the pinned upstream CIR config uses batch 128 / BF16. Document this as an adaptation. For controlled fusion attribution, keep the model/data/training setup fixed except for the fusion module. For comparisons between complete external systems, backbones can differ, but their training data, parameters/resources and evaluation conditions must be explicit.

Use the official split files. The audit records 392 and 419 train/val image overlaps across shirt/toptee categories, despite zero within-category overlap. This is a property to disclose and match across pooled-category experiments; do not silently create a new filtered benchmark and compare its score to published values.

**FashionIQ validation is a legitimate literature comparison split.** SPRC explicitly evaluates validation because test ground truth is unavailable, and QuRe reports FashionIQ validation. Missing local test targets does not prevent a conference-paper comparison. Keep the split label accurate, disclose validation-based tuning/selection and avoid presenting it as untouched test performance. [SPRC, Section 4.1](https://proceedings.iclr.cc/paper_files/paper/2024/file/8204f54f6cd8f0db1765a274e94ceaac-Paper-Conference.pdf), [QuRe, Table 2](https://arxiv.org/html/2507.12416v1).

## 5. Recommended order of work

1. **Use the evidence already available.** Put all three simple FashionCLIP baselines and both final UniFashion scoring variants in the paper's main measured-results table. Report all three categories plus macro. Separate checkpoint progression, tuning and protocol diagnostics.
2. **Resolve the fusion claim.** Re-evaluate the saved standalone gate through the common evaluator and obtain full Recall results for the intended integrated-gate checkpoint. Do not substitute the standalone model for the integrated model.
3. **Add independent published competitors using released weights.** The [official CLIP4CIR repository](https://github.com/ABaldrati/CLIP4Cir) advertises FashionIQ encoder/Combiner checkpoints in Releases; [SPRC's official repository](https://github.com/chunmeifeng/SPRC) lists FashionIQ checkpoints and evaluation commands. This offers an inference-first route before considering expensive training. Downloads and compatibility have not been tested in this audit. Keep separate environments/model adapters and verify checkpoint identities.
4. **Make the comparison auditable.** Match query/gallery coverage, category aggregation, reference removal, text composition, image IDs and file provenance; retain the appropriate model-specific preprocessing. Record auxiliary captions, reranking, pretraining and backbone differences. Add a small multi-method published-reference table with citations and explicit “reported” versus “reproduced here” labels. Do not combine zero-shot and supervised results without identifying their regimes.
5. **Add controls that support the intended contribution.** Before/after fine-tuning, gate versus matched fusion alternatives, caption ablation and paired error/uncertainty analysis are higher priority than another LR sweep.

This recommendation does not require reproducing every method from scratch. It does require enough relevant strong comparisons to support the paper's particular claim. Checkpoint and LR comparisons alone cannot establish a new method's contribution.

## 6. Suggested response to the advisor

> Nhóm đã có so sánh thực nghiệm với ba baseline FashionCLIP: image-only, text-only và additive, cùng ablation có/không cộng score caption gallery cho UniFashion. Tuy nhiên, phần so sánh với các phương pháp CIR mạnh còn hạn chế: hiện chủ yếu đối chiếu số liệu UniFashion trong bài báo, chưa chạy các đối thủ độc lập trên cùng giao thức. Nhóm sẽ bổ sung đầy đủ các baseline vào bảng kết quả, đánh giá checkpoint của các phương pháp như CLIP4CIR và SPRC, và hoàn thiện so sánh UniFashion với attention-fusion dưới cùng điều kiện. Các số liệu trích từ bài báo sẽ được ghi rõ nguồn và tách khỏi số liệu nhóm tự chạy.

GitHub visibility also needs attention: the raw baseline JSON/CSV and current result CSVs are tracked, but most narrative reports and guides, including the current `weekly_report.md`, are still untracked at this audit. A reader can see the raw results yet miss the explanation. Publish the intended narrative files with the paper artifact; this observation does not imply which files the advisor actually reviewed.
