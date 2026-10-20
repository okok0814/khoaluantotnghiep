"""Build the Colab add-on without replacing fingerprinted full-training sources."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    evidence = ROOT / "logs/week41_fashioniq_evaluation"
    output = ROOT / "checkpoints/unifashion_tuning.zip"
    output.parent.mkdir(exist_ok=True)
    names = ["configs/unifashion_tuning.json", "src/unifashion_tuning.py", "src/fashioniq_evaluation.py",
             "scripts/tune_unifashion.py", "scripts/report_unifashion_tuning.py",
             "scripts/evaluate_unifashion_fashioniq.py", "scripts/report_fashioniq_evaluation.py",
             "tests/test_unifashion_tuning.py", "tests/test_fashioniq_evaluation.py"]
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(ROOT / name, name)
        for directory in ("evaluation_data", "baselines"):
            for path in sorted((evidence / directory).iterdir()):
                z.write(path, "evaluation_inputs/" + path.relative_to(evidence).as_posix())
        z.write(evidence / "preparation.json", "evaluation_inputs/preparation.json")
    manifest = {"path": str(output), "bytes": output.stat().st_size,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "replaces_original_training_source": False, "includes_weights_or_images": False}
    report = ROOT / "logs/week42_unifashion_tuning"
    report.mkdir(exist_ok=True)
    (report / "bundle.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
