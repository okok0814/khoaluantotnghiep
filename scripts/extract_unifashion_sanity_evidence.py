"""Extract existing notebook outputs, without claiming to rerun the GPU experiment."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "notebooks/39_17_UniFashion_Finetune_Sanity.ipynb"
    notebook = json.loads(source.read_text(encoding="utf-8"))
    decoder = json.JSONDecoder()
    selected = None
    for cell_index, cell in enumerate(notebook["cells"]):
        stdout = "".join("".join(o.get("text", [])) for o in cell.get("outputs", [])
                         if o.get("output_type") == "stream")
        if "sanity_passed" not in stdout or '"step":' not in stdout:
            continue
        objects, offset = [], 0
        while (start := stdout.find("{", offset)) >= 0:
            try:
                obj, end = decoder.raw_decode(stdout[start:])
                objects.append(obj)
                offset = start + end
            except json.JSONDecodeError:
                offset = start + 1
        statuses = [o for o in objects if o.get("status") == "sanity_passed"]
        rows = [o for o in objects if "step" in o and "loss" in o]
        if statuses and rows:
            selected = cell_index, stdout, statuses[-1], rows
            break
    if selected is None:
        raise ValueError("No completed sanity run with step logs found in notebook")
    index, stdout, status, rows = selected
    if len(rows) != status["optimizer_steps"] or rows[-1]["samples_seen"] != status["train_samples"]:
        raise ValueError("Notebook step logs do not agree with recorded status")
    out = ROOT / "reports/week8_unifashion/prior_sanity"
    out.mkdir(parents=True, exist_ok=True)
    evidence = {"provenance": "Extracted saved notebook stdout; not a new run or independent checkpoint verification",
                "source_notebook": source.relative_to(ROOT).as_posix(),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "cell_index_zero_based": index, "recorded_status": status}
    (out / "notebook_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    (out / "recorded_stdout.log").write_text(stdout, encoding="utf-8")
    with (out / "train_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for key in ("loss", "loss_itc", "loss_itm", "loss_ttc"):
        axes[0].plot([r["step"] for r in rows], [r[key] for r in rows], marker=".", label=key)
    axes[0].set(xlabel="Optimizer update", ylabel="Observed batch loss", title="Prior sanity: 24 train queries")
    axes[0].legend()
    names = ["loss_itc", "loss_itm", "loss_ttc"]
    axes[1].bar([i-.18 for i in range(3)], [status["validation_before"][k] for k in names],
                width=.36, label="Before")
    axes[1].bar([i+.18 for i in range(3)], [status["validation_after"][k] for k in names],
                width=.36, label="After 12 updates")
    axes[1].set(xticks=range(3), xticklabels=["ITC", "ITM", "TTC"], ylabel="Mean loss",
                title="Prior sanity: 12 validation queries")
    axes[1].legend()
    fig.suptitle("Saved notebook evidence — NOT the full FashionIQ fine-tuning run")
    fig.savefig(out / "sanity_curves.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"saved": str(out), "steps": len(rows),
        "total_validation_before": sum(status["validation_before"].values()),
        "total_validation_after": sum(status["validation_after"].values())}, indent=2))


if __name__ == "__main__":
    main()
