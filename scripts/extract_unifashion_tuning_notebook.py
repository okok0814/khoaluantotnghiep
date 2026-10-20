"""Extract saved Colab output as report evidence, without executing notebook code."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd


def stream(cell):
    return "".join("".join(o.get("text", [])) for o in cell.get("outputs", []))


def json_objects(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"(?m)^\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def summaries(text):
    values = {}
    for line in text.splitlines():
        for method in ("unifashion_image_score", "unifashion_image_caption_score"):
            if line.startswith(method + " ["):
                values[method] = ast.literal_eval(line[len(method) + 1:])
    if len(values) != 2:
        raise ValueError("Expected both completed screening summaries")
    for rows in values.values():
        parts = rows[:3]
        if [r["category"] for r in parts] != ["dress", "shirt", "toptee"]:
            raise ValueError("Wrong category coverage")
        if any(r["queries"] != 64 or r["gallery_size"] != 512 for r in parts):
            raise ValueError("This extractor expects the saved 192-query, 512-gallery study")
        macro = next(r for r in rows if r["category"] == "MACRO")
        micro = next(r for r in rows if r["category"] == "MICRO")
        for k in (10, 50):
            metric = f"Recall@{k}"
            expected = sum(r[metric] for r in parts) / 3
            for aggregate in (macro, micro):
                if aggregate["queries"] != 192 or not math.isclose(aggregate[metric], expected, abs_tol=1e-10):
                    raise ValueError("Inconsistent screening aggregate")
            for row in parts:
                hits = row[metric] * row["queries"] / 100
                if not math.isclose(hits, round(hits), abs_tol=1e-10):
                    raise ValueError("Recall inconsistent with integer hit count")
    return values


def extract(notebook, output, config_path):
    notebook, output = Path(notebook), Path(output)
    raw = notebook.read_bytes()
    nb = json.loads(raw)
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    snapshot_cell = next(c for c in code if "'snapshot'" in "".join(c["source"]))
    base_cell = next(c for c in code if "".join(c["source"]).strip() == "evaluate_checkpoint(PARENT, 'base', 'screen')")
    trials_cell = next(c for c in code if "for trial in study['config']['trials']:" in "".join(c["source"]))
    full_cell = next(c for c in code if "evaluate_checkpoint(PARENT, 'base', 'full')" in "".join(c["source"]))
    setup_cell = next(c for c in code if "'tests/test_unifashion_tuning.py'" in "".join(c["source"]))
    parents = [v for v in json_objects(stream(snapshot_cell)) if "checkpoint_sha256" in v]
    if not parents or any(v != parents[0] for v in parents):
        raise ValueError("Missing/inconsistent parent metadata")
    parent = parents[0]
    config = json.loads(Path(config_path).read_text())
    groups = re.split(r"(?m)^Saved trial checkpoint: 0/\d+ updates\s*$", stream(trials_cell))[1:]
    if len(groups) != len(config["trials"]):
        raise ValueError("Trial output does not match the configured study")
    screen = {"base": summaries(stream(base_cell))}
    metric_rows, states = [], []
    for trial, group in zip(config["trials"], groups, strict=True):
        metrics = [v for v in json_objects(group) if "update" in v]
        budget = 3 * config["train_per_category"]
        end = budget // trial["batch_size"]
        if (not metrics or metrics[-1]["update"] != end or
                f"Saved trial checkpoint: {end}/{end} updates" not in group):
            raise ValueError("Missing trial completion evidence")
        if any(v["samples"] != trial["batch_size"] or v["lr"] != trial["learning_rate"] for v in metrics):
            raise ValueError("Logged LR/batch differs from trial mapping")
        screen[trial["name"]] = summaries(group)
        metric_rows += [{"trial": trial["name"], **v} for v in metrics]
        states.append({**trial, "updates": end, "train_examples": budget,
                       "status_in_notebook": "trial_completed", "logged_updates": len(metrics)})
    selections = [v for v in json_objects(stream(trials_cell)) if "selected" in v and "ready" in v]
    if len(selections) != 1 or selections[0]["parent_sha256"] != parent["checkpoint_sha256"]:
        raise ValueError("Selection/parent mismatch")
    selection = selections[0]
    rows = [{"trial": trial, "method": method, **row}
            for trial, methods in screen.items() for method, summary in methods.items() for row in summary]
    df = pd.DataFrame(rows)
    macro = df[df.category == "MACRO"].copy()
    macro["mean_R"] = (macro["Recall@10"] + macro["Recall@50"]) / 2
    primary = macro[macro.method == config["selection_method"]].copy()
    selected = max(primary.to_dict("records"),
                   key=lambda r: (r["mean_R"], r["trial"] == "base", r["trial"]))["trial"]
    if not selection["ready"] or selection["selected"] != selected:
        raise ValueError("Displayed selection inconsistent with reported scores")
    reference = primary[primary.trial == "base"].iloc[0]
    for metric in ("Recall@10", "Recall@50", "mean_R"):
        primary["delta_" + metric + "_pp"] = primary[metric] - reference[metric]
    full_text = stream(full_cell)
    progress_lines = re.findall(r"(?m)^(?:dress|shirt|toptee): (?:gallery|queries) \d+/\d+", full_text)
    if "unifashion_image_caption_score [" in full_text:
        raise ValueError("Full results now present; review separately instead of assuming interruption")
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "screening_all_categories.csv", index=False)
    primary.to_csv(output / "screening_comparison.csv", index=False)
    pd.DataFrame(states).to_csv(output / "trial_completion.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output / "logged_training_updates.csv", index=False)
    for name, cell in (("setup", setup_cell), ("snapshot", snapshot_cell), ("base_screen", base_cell),
                       ("trials_and_screen", trials_cell), ("full_validation_interrupted", full_cell)):
        (output / (name + ".txt")).write_text(stream(cell), encoding="utf-8")
    displays = ["".join(o["data"]["text/markdown"]) for o in full_cell.get("outputs", [])
                if "text/markdown" in o.get("data", {})]
    if displays:
        (output / "report_displayed_before_full_validation.md").write_text(displays[0], encoding="utf-8")
    audit = {"source_notebook": notebook.as_posix(), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "evidence_type": "saved_notebook_outputs_not_original_run_artifacts",
        "parent_metadata_reported": parent, "selection_reported": selection,
        "trial_completion": states, "full_validation_last_saved_line": progress_lines[-1] if progress_lines else None,
        "full_validation_summary_present": False,
        "interruption_reason": "Colab usage exhausted (user report); saved output itself contains no quota error",
        "checks": ["Both screening methods and all three categories present for base/three trials",
                   "Aggregate Recall matches reported per-category percentages",
                   "Trial end updates, LR and batch match console output/config",
                   "Selection recomputed from unrounded reported metrics"],
        "limits": ["No trial checkpoint or original result JSON/per-query CSV received",
                   "Cannot independently recompute target ranks or identify per-query regressions",
                   "Training log contains selected updates only, not full loss history",
                   "Last saved notebook output may precede the actual runtime stop"],
        "extracted_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in output.iterdir() if p.is_file() and p.name != "notebook_output_audit.json"}}
    (output / "notebook_output_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(primary[["trial", "Recall@10", "Recall@50", "mean_R"]].to_string(index=False))
    print("Full validation last saved progress:", audit["full_validation_last_saved_line"])
    return primary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/unifashion_tuning.json"))
    args = parser.parse_args()
    extract(args.notebook, args.output, args.config)
