"""Package a small code ZIP separately from the audited FashionIQ data ZIP."""
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "requirements-unifashion.txt", "configs/unifashion_sanity.json",
    "configs/unifashion_finetune_colab.json",
    "scripts/download_unifashion.py", "scripts/check_unifashion_environment.py",
    "scripts/train_unifashion_sanity.py", "scripts/train_unifashion_finetune.py",
    "scripts/report_unifashion_finetune.py", "scripts/prepare_unifashion_sanity.py",
    "src/unifashion_runtime.py", "src/unifashion_sanity_data.py",
    "src/unifashion_training.py", "src/fashioniq_unifashion_dataset.py",
    "tests/test_unifashion_sanity.py", "tests/test_unifashion_training.py",
]


def main():
    output = ROOT / "checkpoints/unifashion_week8_code.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, name)
    from scripts.prepare_unifashion_sanity import digest
    bundles = {name: {"bytes": (ROOT / "checkpoints" / name).stat().st_size,
                      "sha256": digest(ROOT / "checkpoints" / name)}
               for name in (output.name, "unifashion_week8_data.zip")}
    evidence = ROOT / "reports/week8_unifashion"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "bundles.json").write_text(json.dumps(bundles, indent=2), encoding="utf-8")
    print(json.dumps(bundles, indent=2))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
