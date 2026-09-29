"""Read every real sanity sample through the caption-aware PyTorch loader."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from src.unifashion_sanity_data import UniFashionSanityDataset, collate
from scripts.train_unifashion_sanity import verify_subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "reports/week7_unifashion/data")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/week7_unifashion/local_data_checks.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = json.loads((ROOT / "configs/unifashion_sanity.json").read_text())
    verify_subset(args.data_dir, args.images_dir)
    result = {"checkpoint_forward_tested": False, "splits": {}}
    for split in ("train", "val"):
        dataset = UniFashionSanityDataset(args.data_dir / f"{split}.csv", args.images_dir, config)
        count, batches = 0, 0
        for batch in DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collate, num_workers=0):
            n = len(batch["text_input"])
            assert n >= 2
            for key in ("image", "target"):
                assert batch[key].shape == (n, 3, config["image_size"], config["image_size"])
                assert torch.isfinite(batch[key]).all()
            for key in ("text_input", "reference_caption", "target_caption"):
                assert len(batch[key]) == n and all(batch[key])
            count += n
            batches += 1
        assert count == len(dataset)
        result["splits"][split] = {"samples": count, "batches": batches, "finite_pixels": True,
                                     "captions_present": True}
    result["status"] = "data_checks_passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
