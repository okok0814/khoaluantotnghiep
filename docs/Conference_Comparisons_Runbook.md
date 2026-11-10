# Conference paper comparisons — Colab T4

**Reporting update, 10 November 2026:** the supplied Colab comparison evidence has been verified. See the [Vietnamese weekly report](../logs/paper_comparisons/weekly_report.md) for completed baseline, weighted-additive/MLP, SPRC, CLIP4CIR and UniFashion-initialization results. The setup notes below describe the earlier preparation state; those completed runs do not need to be repeated merely to write the report.

Prepared 4 November 2026. Start with [45_11_Conference_Comparisons.ipynb](../notebooks/45_11_Conference_Comparisons.ipynb). This workflow extends the [comparison audit](Baseline_SOTA_Comparison_Audit.md). It does not restart the completed 90,000-update UniFashion training.

## What has been completed locally

- Re-evaluated the saved **standalone FashionCLIP gate**, with all 6,016 validation queries, complete category galleries, reference exclusion and the shared stable tie rule: macro **38.7645% R@10 / 60.8480% R@50**. Its historical notebook reported 38.7480/60.8480 using a different numeric/ranking path. This small change is not a training improvement.
- Validated and consolidated image-only, text-only, additive, the standalone gate, and both final UniFashion scoring variants. See [comparison_report.md](../logs/paper_comparisons/report/comparison_report.md), [category results](../logs/paper_comparisons/report/measured_results.csv), and [LaTeX table](../logs/paper_comparisons/report/measured_table.tex). The LaTeX table requires `booktabs`.
- Computed 5,000 category-stratified paired query bootstrap replicates. [Paired differences](../logs/paper_comparisons/report/paired_differences.csv) distinguish the gate from text-only and isolate the gallery-caption scoring gain. These intervals condition on the checkpoint/gallery; they do **not** measure training-seed variability or correct for validation-based selection. Queries sharing an image are not independent; interpret narrow intervals cautiously.
- Created [error-review samples](../logs/paper_comparisons/report/error_review_samples.csv) and an offline visual sheet at `reports/paper_comparisons/error_review.html`. Its 120 examples are sampled with a fixed seed from category/outcome groups. Human annotations are still pending; use the sheet's download button to export them.
- Checked the official SPRC and CLIP4CIR source with tiny CPU inference fixtures and tested protocol/resume safeguards. These tests are not full-model benchmark runs. Full downloaded checkpoint loading and T4 execution remain to be verified on Colab.

## Run order

| Stage | Action | Hardware | Why |
|---|---|---|---|
| 1 | Download FashionCLIP assets, then SPRC and CLIP4CIR when needed | Colab CPU | Avoid consuming T4 time on downloads |
| 2 | `features_controls` | T4 | Extract features from exactly the UniFashion image files; run fixed baselines and three seeds of three small fusion controls |
| 3 | `sprc` | T4 | Independent published CIR competitor, released FashionIQ **ViT-L** checkpoint |
| 4 | `clip4cir` | T4 | Independent CLIP4CIR/Combiner **RN50x4 fullft** checkpoint pair |
| 5 | `unifashion-initial` | T4 | Same original domain-pretraining-to-CIR initialization, before FashionIQ fine-tuning |
| 6 | Generate report and export evidence | CPU | Recompute only from complete verified predictions |
| Pending | Evaluate integrated UniFashion + attention-fusion | T4, after teammate supplies checkpoint and training source | User confirmed checkpoint unavailable |

You can split these across several Colab sessions. Run **one GPU task at a time**. There is no requirement to finish all tasks in one session, and no 10-epoch UniFashion retraining in this notebook. T4 runtimes depend on model and disk speed; no unmeasured completion-time estimate is promised.

## Files and paths

The prepared code/evidence bundle is `checkpoints/paper_comparisons.zip`. Upload it to:

```text
/content/drive/MyDrive/Thesis/UniFashion/week8/inputs/paper_comparisons.zip
```

Reuse the existing `inputs/unifashion_week8_data.zip` and `pretrained_assets/` in that same Week8 directory. The new bundle already contains the verified original code ZIP, small pinned source files, tokenizer files, baseline predictions and final UniFashion predictions. It contains no images or model weights.

New results live under:

```text
/content/drive/MyDrive/Thesis/UniFashion/paper_comparisons/
  assets/          # downloaded external checkpoints; not for Git
  features/        # matched FashionCLIP cache; not for Git
  results/         # result JSON/CSV, control checkpoints and resumable caches
  report/          # validated tables, plots, intervals and completion checklist
```

Notebook cells:

1. **Setup and mount Drive:** check paths, unpack into `/content/fashioniq_paper`, verify bundle hashes.
2. **Dependencies and CPU tests:** run all ML commands in new Python processes to avoid the earlier pandas/import-state problem. Keep Colab's working torch/torchvision pair; pin the compatible Transformers stack.
3. **Download the selected assets (CPU is sufficient):** choose `DOWNLOAD_METHOD = 'fashionclip'`, `'sprc'` or `'clip4cir'`. Repeat for methods you plan to evaluate. Downloading assets does not produce a Recall result.
4. **Run one GPU task:** choose `TASK`. Start with `features_controls`. Later use `sprc`, `clip4cir`, then `unifashion-initial`.
5. **Rebuild the comparison report:** safe after any completed task; missing experiments stay pending.
6. **Export evidence:** produces `paper_comparison_evidence.zip` without weights, image data, feature caches or resumable cache shards. Send this ZIP back for report review.

Changing Colab hardware resets `/content`. Rerun cells 1–2 and then the relevant task. Existing downloads/features/results in Drive are reused. Keep the same task name, batch size, software versions and input files when resuming. The scripts reject mixed runs if the run specification changes; choose a new output folder for a deliberate experiment change. Do not bypass a mismatch check by deleting its manifest.

External-model feature extraction saves after each 128 gallery images and query evaluation after 64 queries. An interruption can repeat the unfinished chunk. Fusion controls save model, optimizer and RNG state after every epoch, so a partial epoch repeats. A completed model is never reported from partial predictions.

### SPRC vocabulary-bias initialization error

If Cell 4 fails with `Qformer.cls.predictions.bias` having checkpoint shape `[30523]` and model shape `[30522]`, update to the rebuilt `checkpoints/paper_comparisons.zip`. The original adapter removed this unused language-generation head **after** loading; the shape mismatch occurred before that removal. The corrected adapter removes `Qformer.cls` and excludes its checkpoint keys before loading, while strictly checking all retained retrieval weights, including the token embeddings. SPRC's retrieval functions call `Qformer.bert` directly. The loader records the ignored head keys and any reconstructed deterministic position buffer in its provenance.

Recovery in the current Colab notebook:

1. Replace `week8/inputs/paper_comparisons.zip` in Drive with the updated local bundle.
2. Rerun Cells **1 and 2** to unpack/verify the fix and run its regression tests.
3. Skip Cell 3: the downloaded SPRC assets are reusable. Run Cell **4** with `TASK = 'sprc'`, `BATCH_SIZE = 4`.
4. After `Task finished: sprc`, run Cells **5 and 6** to refresh/export the evidence.

The evaluator preserves a failed initialization directory containing **only** `run_spec.json` as a sibling named `sprc_unstarted_<unique id>` before starting with corrected code. Any existing cache, provenance, result or unknown file prevents this automatic restart; the strict resume check still applies to runs that have started. Keep your completed `features_controls` results. The `_pytree` FutureWarnings are separate from this shape error and do not require a PyTorch downgrade.

CLIP4CIR's [official release](https://github.com/ABaldrati/CLIP4Cir/releases/tag/weights) is split into **8,545,777,978 bytes** across five files. The notebook downloads these to local CPU-runtime disk and streams their concatenation, retaining only the FashionIQ RN50x4 pair in Drive. A CPU runtime reset before extraction may require downloading unfinished parts again. Keep enough local disk space for the parts and weights. The published release digests are pinned in [paper_comparisons.json](../configs/paper_comparisons.json). SPRC's Drive link may be rate-limited by its host; a failed download must not be treated as a completed comparison.

## Experimental controls

All new methods use the existing `evaluation_data/queries.csv`, official galleries and ranking implementation. No targets are used to build query features. Test targets are unavailable locally; label the split **FashionIQ validation**, as is common in the literature.

**Matched pixels.** Existing FashionCLIP embeddings came from another image cache. The new extractor uses the same audited resized JPEG bytes as UniFashion, but retains each encoder's own transform. This removes the unknown-input difference. It does not reproduce published methods' original full-resolution images. The old standalone gate is evaluated on its historical feature cache; new matched-pixel controls are trained afresh.

**Fusion controls.** Frozen FashionCLIP features; all 18,000 official training queries; final epoch selected without looking at validation; 15 epochs, batch 256 with `drop_last`, AdamW LR 1e-3, weight decay 1e-4, temperature 0.07, seeds 42/43/44. The one-way in-batch objective matches the historical gate, including its handling of duplicate targets. Controls are a learned scalar weighted sum, a two-layer residual MLP, and the existing gate. At 512 dimensions, the MLP is within 0.2% of the gate's trainable parameter count. Report all seeds as mean/sample SD; do not select the best seed. This is a frozen-feature control experiment, not a substitute for the integrated UniFashion fusion comparison.

**External methods.** Inference calls official forward implementations from pinned source revisions. CLIP4CIR uses its fine-tuned RN50x4 encoder and matching Combiner. SPRC uses the advertised **ViT-L** checkpoint and its BLIP-2/CLIP-L initialization. Missing active weights or incompatible shapes stop execution. The SPRC ViT-L result must not be labeled as the original paper's full-model 54.92/74.97 row. [Official SPRC repository](https://github.com/chunmeifeng/SPRC), [official CLIP4CIR repository](https://github.com/ABaldrati/CLIP4Cir).

The common protocol feeds the project's period-joined modifier text. The external native evaluators join captions with “and” and do not remove the reference in their FashionIQ ranking functions. Our common-protocol scores therefore need not equal the published scores, even with the same checkpoint. Report them as **released-checkpoint evaluations under our shared protocol**, not exact published-score reproductions. Source revisions, weight hashes, precision and preprocessing are saved with each run.

**UniFashion initialization.** Load the exact original runtime from the protected ZIP, original domain-pretraining checkpoint, seed/config and frozen visual backbone. Some retrieval heads are randomly initialized in the pretraining-to-CIR transfer; the report lists them. This is a before/after initialization control, not a claim that a fully pretrained CIR system was evaluated zero-shot. The current repository runtime injects a gate, so it must not replace this original runtime. Both image-only scoring and image + gallery-caption scoring are exported together.

**Gallery captions.** Existing no-caption-score results already provide an ablation: final UniFashion changes from 45.14/68.53 to 64.79/80.64. Released `next_llava.val.*` captions are extra gallery-side resources; the project has not established equivalence to the paper's caption-generation pipeline. Preserve this distinction. New external controls do not use those captions. Regenerating the paper's caption resources would require a separate generation experiment and is not necessary to report a transparent system comparison.

## Paper tables and claims

Use the generated measured table for methods evaluated here; put [published references](paper_comparison_audit/published_reference_results.csv) in a separate table with citations. The references include UniFashion (EMNLP 2024), SPRC (ICLR 2024), and QuRe (ICML 2025). They are a contextual selection, not a comprehensive November 2026 leaderboard. Check recent related work and the target venue's expectations when finalizing the paper.

Recommended tables/figures:

1. Main results: all three categories plus macro R@10/R@50; image, text, additive, learned controls, two external competitors, UniFashion with/without captions, integrated gate when available.
2. Control study: weighted sum versus MLP versus gate, per seed and mean/SD; separate from query-bootstrap intervals.
3. Before/after fine-tuning and caption scoring ablations, with the random initialization caveat.
4. Separate published-reference table, listing backbone, training regime, caption resources and source table number.
5. Paired gain/loss counts and visually reviewed examples. Describe error types only after reviewing images and text; a labeled-target miss can still retrieve a plausible alternative product.

Do not claim an integrated-fusion gain, completed SOTA reproduction, or superiority to all recent methods while the relevant evidence is pending. Stronger claims require broader competing methods and, for the expensive UniFashion training comparison, matched runs/multiple training seeds if feasible. The current budget prioritizes inference and small control training on T4.

## Checkpoint needed from teammate

Obtain an immutable full-training checkpoint snapshot, its config, code/data/assets fingerprints, training budget/seed, and the exact training source or code ZIP. Evaluate it in that runtime with the same image/caption resources and both scoring variants. Compare to the original run only after recording any differences in initialization, optimizer, batch size, budget and selection. The original 512-d standalone FashionCLIP gate cannot be substituted for the integrated 256-d UniFashion gate.

## Local regeneration and publishing

```powershell
.venv\Scripts\python.exe -B scripts/report_paper_comparisons.py
.venv\Scripts\python.exe -B scripts/package_paper_comparisons.py
```

Commit comparison scripts, tests, config, notebook, this guide and the small `logs/paper_comparisons/` evidence. The ignore rules exclude checkpoint/feature/cache files and the local visual sheet containing dataset images. Do not commit Drive weights or image bundles. Existing historical reports are preserved.

Suggested current-status wording:

> Nhóm đã hợp nhất kết quả của các baseline image-only, text-only, additive, mô hình FashionCLIP với gated fusion và UniFashion sau fine-tune; đồng thời đánh giá lại gated fusion trên cùng giao thức với đầy đủ 6.016 truy vấn validation. Nhóm bổ sung khoảng tin cậy bootstrap theo truy vấn và bộ mẫu phục vụ phân tích lỗi. Các lượt đánh giá checkpoint công bố của CLIP4CIR/SPRC, kiểm soát trước–sau fine-tune và các baseline fusion có ngân sách tương đương đã được chuẩn bị để chạy trên Colab T4; chưa báo cáo chúng là kết quả hoàn tất. So sánh UniFashion tích hợp attention-fusion đang chờ checkpoint của thành viên phụ trách.
