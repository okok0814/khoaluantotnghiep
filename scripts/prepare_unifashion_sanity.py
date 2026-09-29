"""Audit FashionIQ and make deterministic, disjoint, caption-enriched subsets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("dress", "shirt", "toptee")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"candidate", "modifier", "target", "category"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Empty CSV or missing columns: {path}")
    if any(not row[k].strip() for row in rows for k in required):
        raise ValueError(f"Missing train/val fields: {path}")
    return rows


def image_path(root, image_id):
    for folder in (root, *(root / cat for cat in CATEGORIES)):
        for ext in (".jpg", ".jpeg", ".png"):
            path = folder / (image_id + ext)
            if path.is_file():
                return path
    raise FileNotFoundError(f"Image not found: {image_id} under {root}")


def caption_lookup(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    lookup, conflicts = {}, 0
    for row in rows:
        key, caption = row["image"], row["caption"].strip()
        if not caption:
            raise ValueError(f"Empty released caption for {key}")
        if key in lookup and lookup[key] != caption:
            conflicts += 1
        lookup.setdefault(key, caption)  # Stable policy: first released caption per ID.
    return lookup, conflicts


def prepare(csv_dir, images, annotations, upstream, output, config):
    output.mkdir(parents=True, exist_ok=True)
    audit = {"seed": config["seed"], "splits": {}, "sources": {},
             "caption_policy": "first released caption per image ID, within the same split/category"}
    selected = {}
    all_split_ids = {}
    category_split_ids = {}
    for split in ("train", "val"):
        csv_path = csv_dir / f"fashioniq_triplets_{split}.csv"
        rows = read_csv(csv_path)
        audit["sources"][str(csv_path)] = digest(csv_path)
        if any(row["category"] not in CATEGORIES for row in rows):
            raise ValueError("Unknown category in CSV")
        selected[split] = []
        audit["splits"][split] = {"csv_rows": len(rows), "categories": {}}
        all_split_ids[split] = set()
        for category in CATEGORIES:
            official_path = annotations / "captions" / f"cap.{category}.{split}.json"
            split_path = annotations / "image_splits" / f"split.{category}.{split}.json"
            caption_path = upstream / "dataset" / f"next_llava.{split}.{category}.caption.json"
            for path in (official_path, split_path, caption_path):
                audit["sources"][str(path)] = digest(path)
            official = json.loads(official_path.read_text(encoding="utf-8"))
            official_pairs = {(row["candidate"], row["target"]) for row in official}
            official_ids = set(json.loads(split_path.read_text(encoding="utf-8")))
            all_split_ids[split].update(official_ids)
            category_split_ids[(category, split)] = official_ids
            captions, conflicts = caption_lookup(caption_path)
            category_rows = [row for row in rows if row["category"] == category]
            eligible, missing_images, missing_captions, excluded_overlap = [], 0, 0, 0
            for row in category_rows:
                pair = (row["candidate"], row["target"])
                if pair not in official_pairs or not set(pair).issubset(official_ids):
                    raise ValueError(f"CSV row does not belong to official {category}/{split}: {pair}")
                if pair[0] == pair[1]:
                    raise ValueError(f"Self-target triplet: {pair}")
                if split == "val" and set(pair) & all_split_ids["train"]:
                    excluded_overlap += 1
                    continue
                try:
                    for key in pair:
                        image_path(images, key)
                except FileNotFoundError:
                    missing_images += 1
                    continue
                if any(key not in captions for key in pair):
                    missing_captions += 1
                    continue
                eligible.append({**row, "reference_caption": captions[pair[0]],
                                 "target_caption": captions[pair[1]]})
            random.Random(config["seed"] + CATEGORIES.index(category)).shuffle(eligible)
            chosen, targets = [], {row["target"] for row in selected[split]}
            count = config[f"{split}_per_category"]
            for row in eligible:
                if row["target"] in targets:
                    continue
                chosen.append(row)
                targets.add(row["target"])
                if len(chosen) == count:
                    break
            if len(chosen) != count:
                raise ValueError(f"Not enough eligible unique targets: {category}/{split}")
            for row in chosen:
                for key in ("candidate", "target"):
                    with Image.open(image_path(images, row[key])) as image:
                        image.verify()
            selected[split].extend(chosen)
            audit["splits"][split]["categories"][category] = {
                "csv_rows": len(category_rows), "official_triplets": len(official),
                "official_gallery_ids": len(official_ids), "eligible_rows": len(eligible),
                "missing_image_rows": missing_images, "missing_caption_rows": missing_captions,
                "excluded_val_rows_overlapping_any_train_gallery": excluded_overlap,
                "caption_conflicts_first_kept": conflicts, "selected": len(chosen)}
    overlap = all_split_ids["train"] & all_split_ids["val"]
    audit["official_train_val_image_overlap"] = len(overlap)
    audit["official_overlap_by_category"] = {
        f"{a}_train__{b}_val": len(category_split_ids[a, "train"] & category_split_ids[b, "val"])
        for a in CATEGORIES for b in CATEGORIES}
    audit["overlap_policy"] = "Report cross-category official overlap; exclude affected val rows from sanity subset only"
    selected_ids = {split: {row[k] for row in rows for k in ("candidate", "target")}
                    for split, rows in selected.items()}
    audit["selected_train_val_image_overlap"] = len(selected_ids["train"] & selected_ids["val"])
    if audit["selected_train_val_image_overlap"]:
        raise ValueError("Selected train and validation images overlap")
    for split, rows in selected.items():
        path = output / f"{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        audit["splits"][split]["subset_sha256"] = digest(path)
        audit["splits"][split]["selected_rows"] = len(rows)
    audit["selected_images"] = {}
    for rows in selected.values():
        for row in rows:
            for key in ("candidate", "target"):
                path = image_path(images, row[key])
                audit["selected_images"][row[key]] = {"file": path.name, "sha256": digest(path)}
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--csv-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--upstream", type=Path, default=ROOT / "third_party/UniFashion")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/week7_unifashion/data")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/unifashion_sanity.json")
    parser.add_argument("--bundle", type=Path, help="Optional small ZIP of the selected images and CSVs for Colab")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    audit = prepare(args.csv_dir, args.images_dir, args.annotations_dir,
                    args.upstream, args.output, config)
    if args.bundle:
        args.bundle.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name in ("train.csv", "val.csv", "data_audit.json"):
                z.write(args.output / name, f"data/{name}")
            for image_id in audit["selected_images"]:
                path = image_path(args.images_dir, image_id)
                z.write(path, f"images/{path.name}")
        print(f"Bundle: {args.bundle} ({args.bundle.stat().st_size} bytes)")
    print(json.dumps({key: value for key, value in audit.items() if key not in ("sources", "selected_images")}, indent=2))


if __name__ == "__main__":
    main()
