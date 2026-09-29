"""Fail early if Colab's packages differ from the tested UniFashion versions."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import os
import sys


def mismatches(requirements):
    errors = []
    for line in requirements.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, expected = line.split("==")
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = "not installed"
        print(f"{name}: {actual} (expected {expected})", flush=True)
        # CPU/CUDA wheels add a local version suffix, e.g. 2.6.0+cu124.
        if actual.split("+", 1)[0] != expected:
            errors.append(f"{name}=={expected} (found {actual})")
    return errors


def main():
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("HF_HOME", str(root / ".cache/huggingface"))
    errors = mismatches((root / "requirements-unifashion.txt").read_text())
    if errors:
        raise SystemExit("UniFashion environment mismatch:\n  " + "\n  ".join(errors) +
                         "\nRerun the dependency installation cell before tests/training. "
                         "Use the same sys.executable for pip and the runner.")
    import transformers
    from transformers.pytorch_utils import (
        apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer,
    )
    assert all(callable(f) for f in (apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer))
    if transformers.__version__ != version("transformers"):
        raise SystemExit("Imported transformers differs from installed metadata; check sys.path/shadowed modules")
    print(f"Verified in {sys.executable}: {transformers.__file__}", flush=True)


if __name__ == "__main__":
    main()
