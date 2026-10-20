"""Package additive evaluation files; never replace fingerprinted training source."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    evidence = ROOT / "logs/week41_fashioniq_evaluation"
    output = ROOT / "checkpoints/unifashion_recall_eval.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    files = ["src/fashioniq_evaluation.py", "src/unifashion_tuning.py", "scripts/evaluate_unifashion_fashioniq.py",
             "scripts/report_fashioniq_evaluation.py", "tests/test_fashioniq_evaluation.py"]
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for name in files:
            z.write(ROOT / name, name)
        for subdir in ("evaluation_data", "baselines"):
            for p in sorted((evidence / subdir).iterdir()):
                if p.is_file():
                    z.write(p, "evaluation_inputs/" + p.relative_to(evidence).as_posix())
        z.write(evidence / "preparation.json", "evaluation_inputs/preparation.json")
    result = {"path": str(output), "bytes": output.stat().st_size,
              "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "contains_training_weights": False, "contains_images": False}
    (evidence / "bundle.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
