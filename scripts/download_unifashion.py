"""Download official UniFashion assets; model weights stay outside Git.

The default downloads the domain-pretraining checkpoint, tokenizer and the
small upstream source subset. --with-vision also downloads the frozen EVA-G
backbone required to construct the complete model on the GPU machine.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REVISION = "f01a46fb7ed08255fcb52a943d9bcefde03c8eaf"
HF_REVISION = "b13e8e576e24e059df9e16cb8e07457609b92df1"
SOURCE_FILES = [
    "README.md", "cir_ft.sh", "src/blip_fine_tune_2.py", "src/data_utils.py",
    "src/lavis/common/registry.py", "src/lavis/common/dist_utils.py",
    "src/lavis/common/utils.py", "src/lavis/common/logger.py",
    "src/lavis/models/base_model.py", "src/lavis/models/eva_vit.py",
    "src/lavis/models/clip_vit.py", "src/lavis/models/blip2_models/blip2.py",
    "src/lavis/models/blip2_models/Qformer.py",
    "src/lavis/models/blip2_models/blip2_qformer_cir_rerank.py",
    "src/lavis/models/blip_models/blip_outputs.py",
    "src/lavis/configs/models/blip2/blip2_pretrain.yaml",
] + [f"dataset/next_llava.{split}.{cat}.caption.json"
     for split in ("train", "val") for cat in ("dress", "shirt", "toptee")]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path, expected_sha=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected_sha is None or sha256(path) == expected_sha):
        print(f"Already present: {path}", flush=True)
    else:
        temporary = path.with_suffix(path.suffix + ".part")
        print(f"Downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
            size = int(response.headers.get("Content-Length", 0))
            copied, last = 0, time.monotonic()
            while chunk := response.read(4 * 1024 * 1024):
                out.write(chunk)
                copied += len(chunk)
                if time.monotonic() - last > 15:
                    print(f"  {copied / 1e6:.1f} / {size / 1e6:.1f} MB", flush=True)
                    last = time.monotonic()
        if expected_sha and sha256(temporary) != expected_sha:
            raise RuntimeError(f"SHA-256 mismatch: {temporary}")
        temporary.replace(path)
    return {"path": str(path.relative_to(ROOT)), "url": url,
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-vision", action="store_true")
    parser.add_argument("--local-source", type=Path,
                        help="Reuse this clean checkout of the pinned official source")
    args = parser.parse_args()
    assets = ROOT / "checkpoints/unifashion"
    assets.mkdir(parents=True, exist_ok=True)
    manifest_path = assets / "download_manifest.json"
    revision = HF_REVISION
    manifest = {"upstream_revision": UPSTREAM_REVISION, "checkpoint_revision": revision,
                "files": []}

    def record(item):
        manifest["files"].append(item)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    record(download(
        f"https://huggingface.co/UniFashion/UniFashion/resolve/{revision}/pretrain/saved_models/none_lora_0.pt",
        assets / "pretrain/none_lora_0.pt",
        "f272d3a3cc91ae809f844d3e258f6a872be405cb203a550c15c4510c7280b541"))
    for filename in SOURCE_FILES:
        destination = ROOT / "third_party/UniFashion" / filename
        url = f"https://raw.githubusercontent.com/xiangyu-mm/UniFashion/{UPSTREAM_REVISION}/{filename}"
        if args.local_source:
            source = args.local_source / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        record(download(url, destination))
    bert_revision = "86b5e0934494bd15c9632b12f734a8a67f723594"
    for filename in ("config.json", "vocab.txt", "tokenizer_config.json"):
        record(download(f"https://huggingface.co/google-bert/bert-base-uncased/resolve/{bert_revision}/{filename}",
                        assets / "bert-base-uncased" / filename))
    if args.with_vision:
        record(download("https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth",
                        assets / "eva_vit_g.pth"))
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
