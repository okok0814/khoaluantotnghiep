"""Package comparison code and evidence for Colab; no model weights or images."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fashioniq_evaluation import digest
from src.paper_comparison import write_json
from scripts.download_unifashion import SOURCE_FILES


def main():
    names = ["src/paper_comparison.py", "src/paper_model_adapters.py", "src/fashioniq_evaluation.py",
             "src/fashioniq_completion.py", "src/attention_fusion.py", "src/unifashion_sanity_data.py",
             "src/fashioniq_unifashion_dataset.py", "scripts/evaluate_unifashion_final.py",
             "scripts/run_fashionclip_controls.py", "scripts/extract_paper_fashionclip.py",
             "scripts/prepare_paper_models.py", "scripts/evaluate_paper_models.py",
             "scripts/report_paper_comparisons.py", "scripts/export_paper_comparisons.py",
             "scripts/build_paper_error_review.py",
             "configs/paper_comparisons.json", "tests/test_paper_comparisons.py",
             "logs/fashionclip_text_metadata.csv", "checkpoints/unifashion_week8_code.zip",
             "docs/Conference_Comparisons_Runbook.md"]
    content = {name: ROOT / name for name in names}
    for prefix in ("logs/week41_fashioniq_evaluation/baselines", "logs/week41_fashioniq_evaluation/evaluation_data",
                   "logs/week43_44_unifashion_completion/evidence_completed_20261102/final_results",
                   "logs/paper_comparisons/legacy_gate", "docs/paper_comparison_audit"):
        for path in (ROOT/prefix).glob("*"):
            if path.suffix in (".csv", ".json"):
                content[path.relative_to(ROOT).as_posix()] = path
    for name in SOURCE_FILES:
        if not name.startswith("dataset/"):
            content["third_party/UniFashion/"+name] = ROOT / "third_party/UniFashion" / name
    # Small official source copies support CPU adapter tests; no external weights.
    for model in ("clip4cir", "sprc"):
        source = ROOT / ".cache/paper_assets" / model / "source"
        for path in source.rglob("*"):
            if path.is_file() and not path.name.endswith(".download.json") and "__pycache__" not in path.parts:
                content[path.relative_to(ROOT).as_posix()] = path
    for path in (ROOT / "checkpoints/unifashion/bert-base-uncased").glob("*"):
        if path.suffix in (".json", ".txt"):
            content[path.relative_to(ROOT).as_posix()] = path
    reference = json.loads((ROOT / "logs/week43_44_unifashion_completion/evidence_completed_20261102/final_results/unifashion_image_score.json").read_text())
    with zipfile.ZipFile(ROOT / "checkpoints/unifashion_week8_code.zip") as z:
        for name, sha in reference["provenance"]["training_fingerprint"]["code"].items():
            matches = [n for n in z.namelist() if Path(n).name == name]
            if len(matches) != 1 or hashlib.sha256(z.read(matches[0])).hexdigest() != sha:
                raise ValueError("Original source ZIP changed: " + name)
    manifest = {"files": {name: digest(path) for name,path in sorted(content.items())},
                "contains_model_weights": False, "contains_images": False,
                "original_model_runtime_loaded_only_from_verified_zip": True}
    output = ROOT / "checkpoints/paper_comparisons.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for name,path in content.items():
            z.write(path,name)
        z.writestr("paper_bundle_manifest.json",json.dumps(manifest,indent=2))
    receipt = {"file": output.name, "bytes": output.stat().st_size, "sha256": digest(output), **manifest}
    write_json(ROOT / "logs/paper_comparisons/bundle.json", receipt)
    print(json.dumps({k:receipt[k] for k in ("file","bytes","sha256")},indent=2))


if __name__ == "__main__":
    main()
