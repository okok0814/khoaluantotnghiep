"""Create one small upload ZIP; excludes pretrained weights and private PDFs."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "requirements-unifashion.txt", "configs/unifashion_sanity.json",
    "scripts/download_unifashion.py", "scripts/train_unifashion_sanity.py",
    "scripts/check_unifashion_environment.py",
    "scripts/prepare_unifashion_sanity.py", "src/unifashion_runtime.py",
    "src/unifashion_sanity_data.py", "src/fashioniq_unifashion_dataset.py",
    "tests/test_unifashion_sanity.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-bundle", type=Path, default=ROOT / "checkpoints/unifashion_sanity_data.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/unifashion_week7_colab.zip")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for filename in FILES:
            out.write(ROOT / filename, filename)
        with zipfile.ZipFile(args.data_bundle) as data:
            for name in data.namelist():
                out.writestr(name, data.read(name))
    print(f"Ready to upload: {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
