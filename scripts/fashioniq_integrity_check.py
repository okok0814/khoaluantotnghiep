
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

FASHIONIQ_DIR = "data/dataset3/fashion-iq"
METADATA_DIR = "data/dataset3/fashion-iq-metadata"
CATEGORIES = ["dress", "shirt", "toptee"]
SPLITS = ["train", "val", "test"]


def load_split_counts():
    """Đếm số lượng ID ảnh theo từng split"""
    counts = {}
    for cat in CATEGORIES:
        for part in SPLITS:
            path = f"{FASHIONIQ_DIR}/image_splits/split.{cat}.{part}.json"
            ids = json.load(open(path))
            counts[(cat, part)] = len(ids)
    return counts


def load_caption_counts():
    """Đếm số cặp (candidate, target, caption)."""
    counts = {}
    for cat in CATEGORIES:
        for part in SPLITS:
            path = f"{FASHIONIQ_DIR}/captions/cap.{cat}.{part}.json"
            if os.path.exists(path):
                data = json.load(open(path))
                counts[(cat, part)] = len(data)
    return counts


def load_asin2url():
    """Tải bảng ánh xạ ASIN -> URL ảnh gốc trên Amazon."""
    mapping = {}
    for cat in CATEGORIES:
        path = f"{METADATA_DIR}/image_url/asin2url.{cat}.txt"
        mapping[cat] = {}
        for line in open(path, encoding="utf-8", errors="ignore"):
            parts = line.strip().split("\t")
            if len(parts) == 2:
                mapping[cat][parts[0].strip()] = parts[1].strip()
    return mapping


def check_mapping_coverage(asin2url):
    """mọi ID trong split có URL tương ứng không?"""
    report = {}
    for cat in CATEGORIES:
        all_ids = set()
        for part in SPLITS:
            path = f"{FASHIONIQ_DIR}/image_splits/split.{cat}.{part}.json"
            all_ids.update(json.load(open(path)))
        have_url = sum(1 for i in all_ids if i in asin2url[cat])
        report[cat] = {
            "total_referenced": len(all_ids),
            "have_url_entry": have_url,
            "missing_url_entry": len(all_ids) - have_url,
        }
    return report


def check_url_liveness(asin2url, sample_size=300, max_workers=20, timeout=5):
    """lấy mẫu ngẫu nhiên các URL và thử HEAD request xem còn sống không"""
    import requests

    results = {}
    for cat in CATEGORIES:
        all_urls = list(asin2url[cat].items())
        sample = random.sample(all_urls, min(sample_size, len(all_urls)))

        alive, dead, errors = 0, 0, 0

        def _check(item):
            asin, url = item
            try:
                r = requests.head(url, timeout=timeout, allow_redirects=True)
                return asin, r.status_code == 200
            except Exception:
                return asin, None

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_check, item) for item in sample]
            for f in as_completed(futures):
                asin, ok = f.result()
                if ok is True:
                    alive += 1
                elif ok is False:
                    dead += 1
                else:
                    errors += 1

        results[cat] = {
            "sample_size": len(sample),
            "alive": alive,
            "dead_or_404": dead,
            "request_errors": errors,
            "estimated_live_pct": round(alive / len(sample) * 100, 2),
        }
    return results


if __name__ == "__main__":
    print("=== 1. Split counts ===")
    split_counts = load_split_counts()
    total = 0
    for (cat, part), n in split_counts.items():
        print(f"  {cat:8s} {part:6s}: {n}")
        total += n
    print(f"  TOTAL: {total}")

    print("\n=== 2. Caption pair counts ===")
    cap_counts = load_caption_counts()
    total_caps = sum(cap_counts.values())
    for (cat, part), n in cap_counts.items():
        print(f"  {cat:8s} {part:6s}: {n}")
    print(f"  TOTAL: {total_caps}")

    print("\n=== 3. ASIN->URL mapping coverage ===")
    asin2url = load_asin2url()
    coverage = check_mapping_coverage(asin2url)
    for cat, r in coverage.items():
        print(f"  {cat:8s}: {r['total_referenced']} referenced | "
              f"{r['have_url_entry']} have URL | {r['missing_url_entry']} missing")

    print("\n=== 4. URL liveness sample check ===")
    print("  # liveness = check_url_liveness(asin2url, sample_size=300)")
    print("  # print(liveness)")
    liveness = check_url_liveness(asin2url, sample_size=300)
    print(liveness)
