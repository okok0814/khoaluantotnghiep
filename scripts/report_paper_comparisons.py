"""Validate complete prediction artifacts and build measured tables, paired CIs and review samples."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import numpy as np
import pandas as pd
from src.fashioniq_completion import read_result
from src.fashioniq_evaluation import digest, protocol
from src.paper_comparison import KEYS, paired_bootstrap, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--new-results", type=Path, default=ROOT / "logs/paper_comparisons")
    p.add_argument("--output", type=Path, default=ROOT / "logs/paper_comparisons/report")
    p.add_argument("--replicates", type=int, default=5000)
    args = p.parse_args()
    base = args.root / "logs/week41_fashioniq_evaluation"
    final = args.root / "logs/week43_44_unifashion_completion/evidence_completed_20261102/final_results"
    eval_data = base / "evaluation_data"
    queries, _, fingerprint = protocol(eval_data)
    entries = [(base / "baselines", "fashionclip_"+m, "FashionCLIP " + m.replace("_", " "), "legacy pixels", None)
               for m in ("image_only", "text_only", "additive")]
    entries += [(final, "unifashion_" + m, label, "audited resized pixels", "original") for m, label in
                (("image_score", "UniFashion 90k / image score"),
                 ("image_caption_score", "UniFashion 90k / image + caption score"))]
    new = args.new_results
    if new.exists():
        for path in sorted(new.rglob("*_per_query.csv")):
            if args.output.resolve() in path.resolve().parents:
                continue
            method = path.name.removesuffix("_per_query.csv")
            if not path.with_name(method+".json").exists():
                continue
            entries.append((path.parent, method, method.replace("_", " "), "see run provenance", None))
    args.output.mkdir(parents=True, exist_ok=True)
    tables, ci, frames, hashes, labels = [], [], {}, {}, {}
    for folder, method, label, pixels, variant in entries:
        if method in frames:
            raise ValueError("Ambiguous duplicate method; separate experiment roots: " + method)
        result = read_result(folder, method, eval_data, variant=variant, expected_steps=90000 if variant else None)
        frame = pd.read_csv(folder / (method+"_per_query.csv"), keep_default_na=False)
        frames[method], labels[method] = frame, label
        for extension in (".json", "_per_query.csv"):
            path = folder / (method+extension)
            hashes[str(path)] = digest(path)
        for row in result["summary"]:
            tables.append({"method": method, "label": label, "image_inputs": pixels, **row})
        for interval in paired_bootstrap(frame, repeats=args.replicates):
            ci.append({"method": method, **interval})
    table = pd.DataFrame(tables)
    table.to_csv(args.output / "measured_results.csv", index=False)
    pd.DataFrame(ci).to_csv(args.output / "query_bootstrap_ci.csv", index=False)
    pairs = [("fashionclip_text_only", "fashionclip_additive"),
             ("fashionclip_legacy_gate", "fashionclip_additive"),
             ("fashionclip_legacy_gate", "fashionclip_text_only"),
             ("unifashion_image_score", "fashionclip_text_only"),
             ("unifashion_image_caption_score", "unifashion_image_score"),
             ("unifashion_image_score", "unifashion_initial_image_score"),
             ("unifashion_image_caption_score", "unifashion_initial_image_caption_score")]
    for seed in (42, 43, 44):
        pairs += [(f"fashionclip_gate_seed{seed}", f"fashionclip_{control}_seed{seed}")
                  for control in ("mlp", "weighted")]
    differences, transitions, review = [], [], []
    for a, b in pairs:
        if a not in frames or b not in frames:
            continue
        for row in paired_bootstrap(frames[a], frames[b], repeats=args.replicates):
            differences.append({"system_A": a, "system_B": b, "difference": "A minus B (percentage points)", **row})
        left = frames[a].set_index(KEYS).sort_index()
        right = frames[b].set_index(KEYS).sort_index()
        joined = left[["text", "target_rank", "top50"]].join(right[["target_rank", "top50"]], lsuffix="_A", rsuffix="_B").reset_index()
        for cat in ("dress", "shirt", "toptee"):
            rows = joined[joined.category == cat]
            for k in (10, 50):
                ahit, bhit = rows.target_rank_A <= k, rows.target_rank_B <= k
                for outcome, mask in (("both_hit", ahit & bhit), ("A_only", ahit & ~bhit),
                                      ("B_only", ~ahit & bhit), ("both_miss", ~ahit & ~bhit)):
                    transitions.append({"system_A": a, "system_B": b, "category": cat,
                                        "k": k, "outcome": outcome, "queries": int(mask.sum())})
                    # Deterministic random exemplars per category/outcome; not a claimed error taxonomy.
                    if k == 10:
                        for row in rows[mask].sample(n=min(2, int(mask.sum())), random_state=42).to_dict("records"):
                            review.append({"system_A": a, "system_B": b, "outcome": outcome, **row,
                                           "human_error_type": "", "reviewer_note": ""})
    pd.DataFrame(differences).to_csv(args.output / "paired_differences.csv", index=False)
    pd.DataFrame(transitions).to_csv(args.output / "query_transitions.csv", index=False)
    pd.DataFrame(review).to_csv(args.output / "error_review_samples.csv", index=False)
    # Training seed spread is distinct from query bootstrap uncertainty.
    seed_rows = []
    for family in ("gate", "mlp", "weighted"):
        rows = table[(table.category == "MACRO") & table.method.str.match(f"fashionclip_{family}_seed[0-9]+$")]
        if len(rows):
            seed_rows.append({"family": family, "seeds": len(rows), **{
                f"{metric}_{stat}": float(rows[metric].mean() if stat == "mean" else rows[metric].std(ddof=1))
                for metric in ("Recall@10", "Recall@50") for stat in ("mean", "sample_std")}})
    pd.DataFrame(seed_rows).to_csv(args.output / "training_seed_summary.csv", index=False)
    columns = ["dress", "shirt", "toptee", "MACRO"]
    lines = ["# Conference comparison evidence", "", "Full FashionIQ validation: 6,016 queries; official category galleries; reference excluded; stable ties. Percentages below are measured here, not copied from papers.", "",
             "| Model | Dress R@10/50 | Shirt R@10/50 | Toptee R@10/50 | Macro R@10/50 |", "|---|---:|---:|---:|---:|"]
    latex = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule", r"Model & \multicolumn{2}{c}{Dress} & \multicolumn{2}{c}{Shirt} & \multicolumn{2}{c}{Toptee} & \multicolumn{2}{c}{Macro} \\"]
    for method, label in labels.items():
        rows = table[table.method == method].set_index("category")
        values = [f"{rows.loc[c, 'Recall@10']:.2f}/{rows.loc[c, 'Recall@50']:.2f}" for c in columns]
        lines.append("| " + " | ".join([label]+values) + " |")
        latex.append(label.replace("_", r"\_").replace("&", r"\&") + " & " + " & ".join(v.replace("/", " & ") for v in values) + r" \\")
    latex += [r"\bottomrule", r"\end{tabular}"]
    (args.output / "measured_table.tex").write_text("\n".join(latex), encoding="utf-8")
    lines += ["", "## Interpretation and limits", "",
              "The historical standalone FashionCLIP gate has now been re-evaluated through the common protocol. It is a different model from the pending integrated UniFashion gate. Small differences from the older gate notebook can arise from precision/normalization/tie handling; do not describe them as training gains.", "",
              "Legacy FashionCLIP pixels remain unverified against the resized UniFashion bundle. Prefer the matched-pixel rows after the new feature extraction finishes. All systems still share the exact query IDs, raw text, galleries and metric implementation.", "",
              "95% percentile intervals use 5,000 category-stratified paired query bootstrap draws by default. They condition on a fixed checkpoint and gallery, ignore dependence between queries sharing images, and do not estimate training-seed variability. Small differences should also be checked across training seeds. Validation was used in previous tuning/checkpoint selection, so these are not untouched test estimates.", "",
              "Published references remain in docs/paper_comparison_audit/published_reference_results.csv; their backbone, training and caption resources can differ. The final UniFashion caption gain is a scoring/resource ablation. The released gallery captions have not been regenerated by this project, so this does not establish a SOTA improvement.", "",
              "error_review_samples.csv contains reproducibly sampled successes/failures, with human labels intentionally blank. Review the images before describing color, material, style or multi-attribute errors.", "",
              "## Outstanding evidence", ""]
    pending = {
        "matched_pixels_and_three_seed_controls": all(f"fashionclip_{family}_seed{s}" in frames for family in ("weighted", "mlp", "gate") for s in (42,43,44)),
        "CLIP4CIR_released_weights": "clip4cir_combiner" in frames,
        "SPRC_released_weights": "sprc_vitl" in frames,
        "UniFashion_initialization_control": "unifashion_initial_image_score" in frames,
        "integrated_UniFashion_gate": False,
        "human_error_annotation": False,
        "caption_generation_equivalence_to_paper": False,
    }
    lines += [f"- {key}: {'results available' if done else 'pending'}" for key, done in pending.items()]
    (args.output / "comparison_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(args.output / "verification.json", {"protocol": fingerprint, "input_sha256": hashes,
        "completed_methods": list(frames), "queries_per_method": len(queries), "bootstrap_replicates": args.replicates,
        "remaining_work": pending, "all_comparisons_complete": all(pending.values())})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    macro = table[table.category == "MACRO"]
    fig, ax = plt.subplots(figsize=(10, max(4, len(macro)*.5)))
    y = np.arange(len(macro))
    ax.barh(y-.17, macro['Recall@10'], .34, label="R@10")
    ax.barh(y+.17, macro['Recall@50'], .34, label="R@50")
    ax.set(yticks=y, yticklabels=macro.label, xlabel="Macro Recall (%)", xlim=(0,100))
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(args.output / ("measured_recall."+extension), dpi=200)
    plt.close(fig)
    print(f"Validated {len(frames)} complete methods; report: {args.output}")


if __name__ == "__main__":
    main()
