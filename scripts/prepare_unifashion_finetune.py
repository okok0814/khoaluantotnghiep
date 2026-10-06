"""Audit ALL train/val triplets and package real FashionIQ for the GPU run."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_unifashion_sanity import CATEGORIES, caption_lookup, digest, image_path, read_csv
from PIL import Image


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(images, annotations, upstream, csv_dir, output, monitor_per_category=32, seed=42):
    output.mkdir(parents=True, exist_ok=False)
    (output / "galleries").mkdir()
    audit = {"seed": seed, "sources": {}, "splits": {}, "files": {},
             "protocol": "All official train/val rows retained; disjoint fixed monitor subset for loss only"}
    enriched, split_ids, required_images = {}, {}, set()
    for split in ("train", "val"):
        csv_path = csv_dir / f"fashioniq_triplets_{split}.csv"
        rows = read_csv(csv_path)
        audit["sources"][str(csv_path)] = digest(csv_path)
        enriched[split] = []
        audit["splits"][split] = {}
        for cat in CATEGORIES:
            original = annotations / "captions" / f"cap.{cat}.{split}.json"
            gallery = annotations / "image_splits" / f"split.{cat}.{split}.json"
            captions_path = upstream / "dataset" / f"next_llava.{split}.{cat}.caption.json"
            for path in (original, gallery, captions_path):
                audit["sources"][str(path)] = digest(path)
            official = json.loads(original.read_text())
            category_rows = [row for row in rows if row["category"] == cat]
            pairs = lambda entries: Counter((row["candidate"], row["target"]) for row in entries)
            if pairs(category_rows) != pairs(official):
                raise ValueError(f"CSV does not contain exactly the official {split}/{cat} pairs")
            split_ids[cat, split] = set(json.loads(gallery.read_text()))
            captions, conflicts = caption_lookup(captions_path)
            if conflicts:
                raise ValueError(f"Conflicting released captions: {captions_path}")
            for row in category_rows:
                ids = {row["candidate"], row["target"]}
                if len(ids) < 2 or not ids.issubset(split_ids[cat, split]):
                    raise ValueError(f"Invalid split IDs: {row}")
                enriched[split].append({**row, "reference_caption": captions[row["candidate"]],
                                        "target_caption": captions[row["target"]]})
                required_images.update(ids)
            if split == "val":
                shutil.copyfile(gallery, output / "galleries" / gallery.name)
                required_images.update(split_ids[cat, split])
            audit["splits"][split][cat] = {"triplets": len(category_rows),
                "gallery_size": len(split_ids[cat, split]),
                "repeated_target_rows": len(category_rows) - len({row["target"] for row in category_rows})}
        if len(enriched[split]) != len(rows):
            raise ValueError("Unknown categories in CSV")
        write_rows(output / f"{split}.csv", enriched[split])
    audit["official_overlap_by_category"] = {
        f"{a}_train__{b}_val": len(split_ids[a, "train"] & split_ids[b, "val"])
        for a in CATEGORIES for b in CATEGORIES}
    train_gallery = set().union(*(split_ids[c, "train"] for c in CATEGORIES))
    monitor, used_targets = [], set()
    for cat in CATEGORIES:
        eligible = [r for r in enriched["val"] if r["category"] == cat
                    and not {r["candidate"], r["target"]} & train_gallery]
        random.Random(seed + CATEGORIES.index(cat)).shuffle(eligible)
        count = 0
        for row in eligible:
            if row["target"] in used_targets:
                continue
            used_targets.add(row["target"])
            monitor.append(row)
            count += 1
            if count == monitor_per_category:
                break
        if count != monitor_per_category:
            raise ValueError("Not enough monitor queries")
    write_rows(output / "monitor.csv", monitor)
    audit["monitor"] = {"rows": len(monitor), "per_category": monitor_per_category,
                         "overlap_with_train_gallery": 0, "metric": "fixed-batch validation loss; not Recall@K"}
    manifest = {}
    for i, image_id in enumerate(sorted(required_images), 1):
        path = image_path(images, image_id)
        with Image.open(path) as image:
            image.convert("RGB").load()
        manifest[image_id] = {"file": path.name, "sha256": digest(path)}
        if i % 5000 == 0:
            print(f"Decoded and hashed {i}/{len(required_images)} images", flush=True)
    (output / "images_manifest.json").write_text(json.dumps(manifest, indent=2))
    for path in sorted(output.rglob("*")):
        if path.is_file():
            audit["files"][path.relative_to(output).as_posix()] = digest(path)
    audit["decoded_images"] = len(manifest)
    audit["status"] = "data_prepared"
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--annotations-dir", type=Path, required=True)
    p.add_argument("--upstream", type=Path, default=ROOT / "third_party/UniFashion")
    p.add_argument("--csv-dir", type=Path, default=ROOT / "data/processed")
    p.add_argument("--output", type=Path, default=ROOT / "checkpoints/week8_data")
    p.add_argument("--bundle", type=Path, default=ROOT / "checkpoints/unifashion_week8_data.zip")
    args = p.parse_args()
    audit, manifest = prepare(args.images_dir, args.annotations_dir, args.upstream, args.csv_dir, args.output)
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(args.output.rglob("*")):
            if path.is_file():
                z.write(path, "data/" + path.relative_to(args.output).as_posix())
        for image_id, record in manifest.items():
            z.write(image_path(args.images_dir, image_id), "images/" + record["file"])
    evidence = ROOT / "reports/week8_unifashion"
    evidence.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.output / "data_audit.json", evidence / "data_audit.json")
    print(json.dumps({"bundle": str(args.bundle), "bytes": args.bundle.stat().st_size,
                      "train_rows": sum(r["triplets"] for r in audit["splits"]["train"].values()),
                      "val_rows": sum(r["triplets"] for r in audit["splits"]["val"].values()),
                      "decoded_images": audit["decoded_images"]}, indent=2))


if __name__ == "__main__":
    main()
