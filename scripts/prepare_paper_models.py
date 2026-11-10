"""Fetch pinned inference sources/assets. Large weights are intended for Colab CPU sessions."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fashioniq_evaluation import digest
from src.paper_comparison import write_json


def fetch(url, path, sha=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_suffix(path.suffix + ".download.json")
    if path.exists():
        actual = digest(path)
        expected = sha or (json.loads(receipt.read_text())["sha256"] if receipt.exists() else None)
        if expected != actual:
            raise ValueError(f"Existing asset is unverified or changed: {path}")
        return path
    part = path.with_suffix(path.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "FashionIQ-paper-comparison",
                                                   **({"Range": f"bytes={offset}-"} if offset else {})})
    with urllib.request.urlopen(request, timeout=120) as response:
        append = offset and response.status == 206
        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("Download server returned the wrong byte range")
        with part.open("ab" if append else "wb") as stream:
            last = time.monotonic()
            while chunk := response.read(4 * 1024 * 1024):
                stream.write(chunk)
                if time.monotonic()-last > 15:
                    print(f"{path.name}: {stream.tell()/1e6:.0f} MB", flush=True)
                    last = time.monotonic()
    actual = digest(part)
    if sha and actual != sha:
        raise ValueError(f"Download checksum mismatch: {part}; remove this partial file before retrying")
    part.replace(path)
    write_json(receipt, {"url": url, "sha256": actual, "bytes": path.stat().st_size,
                         "publisher_hash_available": sha is not None})
    return path


class PartReader:
    """Read concatenated release parts without a second 8.5 GB archive on disk."""
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def read(self, size):
        blocks = []
        while size > 0:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = Path(path).open("rb")
            block = self.current.read(size)
            if not block:
                self.current.close()
                self.current = None
                continue
            blocks.append(block)
            size -= len(block)
        return b"".join(blocks)

    def close(self):
        if self.current is not None:
            self.current.close()


def extract_selected(archive, destination):
    """Extract only regular FashionIQ/RN50x4_fullft files; reject paths/links outside the root."""
    destination = Path(destination).resolve()
    with tarfile.open(fileobj=archive, mode="r|gz") as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if ".." in parts or PurePosixPath(member.name).is_absolute():
                raise ValueError("Unsafe archive path")
            if "FashionIQ" not in parts or "RN50x4_fullft" not in parts or not member.isfile():
                continue
            relative = Path(*parts[parts.index("FashionIQ"):])
            target = destination / relative
            if destination not in target.resolve().parents:
                raise ValueError("Archive escapes destination")
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.with_suffix(target.suffix+".tmp").open("wb") as out:
                shutil.copyfileobj(source, out)
            target.with_suffix(target.suffix+".tmp").replace(target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["clip4cir", "sprc", "fashionclip"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sources-only", action="store_true")
    p.add_argument("--download-cache", type=Path,
                   help="Store large split archives here; only extracted FashionIQ weights go in output")
    args = p.parse_args()
    config = json.loads((ROOT / "configs/paper_comparisons.json").read_text())[args.model]
    output = args.output / args.model
    if args.model == "fashionclip":
        from huggingface_hub import snapshot_download, HfApi
        revision = HfApi().model_info(config["model"], revision=config["revision"]).sha
        location = snapshot_download(config["model"], revision=revision,
                                     allow_patterns=["*.json", "*.txt", "model.safetensors"])
        write_json(output / "manifest.json", {"model": config["model"], "revision": revision,
                   "cache": location, "weights_downloaded": True})
        print("FashionCLIP cached for later extraction:", location, flush=True)
        return
    for item in config["files"]:
        name = item if isinstance(item, str) else item["path"]
        url = f"https://raw.githubusercontent.com/{config['repository']}/{config['revision']}/{name}"
        path = fetch(url, output / "source" / name)
        if isinstance(item, dict):
            content = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest()
            if blob != item["git_blob"]:
                raise ValueError("Pinned Git blob differs: " + name)
    if not args.sources_only:
        if args.model == "clip4cir":
            if (output / "checkpoints.json").exists():
                for record in json.loads((output / "checkpoints.json").read_text()).values():
                    if digest(output / record["path"]) != record["sha256"]:
                        raise ValueError("Existing CLIP4CIR checkpoint changed")
                paths = []
            else:
                cache = args.download_cache or output / "downloads"
                paths = [fetch(a["url"], cache / a["name"], a["sha256"]) for a in config["assets"]]
            if paths:
                archive = PartReader(paths)
                try:
                    extract_selected(archive, output)
                finally:
                    archive.close()
            variant = output / config["checkpoint_variant"]
            found = {key: sorted(variant.rglob(pattern)) for key, pattern in
                     (("clip", "fiq*_clip_*"), ("combiner", "fiq*_comb_*"))}
            if any(len(v) != 1 for v in found.values()):
                raise ValueError(f"Expected one matching encoder/Combiner pair; inspect {variant}: {found}")
            write_json(output / "checkpoints.json", {k: {"path": str(v[0].relative_to(output)), "sha256": digest(v[0])}
                                                       for k, v in found.items()})
        else:
            for name, url in (("blip2_pretrained_vitL.pth", config["base_url"]),
                              ("clip_vit_L.pth", config["vision_url"])):
                fetch(url, output / name)
            checkpoint = output / "sprc_fiq_vitl.pt"
            receipt = output / "sprc_fiq_vitl.pt.download.json"
            if not receipt.exists():
                if checkpoint.exists():
                    raise ValueError("Unverified SPRC checkpoint exists without a download receipt")
                import gdown
                temporary = checkpoint.with_suffix(".part")
                gdown.download(config["checkpoint_url"], str(temporary), fuzzy=True, resume=True)
                # Validate tensor-only checkpoint content before recording a first-use hash.
                import torch
                state = torch.load(temporary, map_location="cpu", weights_only=True)
                if "Blip2QformerCirAlignPrompt" not in state:
                    raise ValueError("SPRC download is not the expected official model checkpoint")
                del state
                temporary.replace(checkpoint)
                write_json(receipt, {"url": config["checkpoint_url"], "sha256": digest(checkpoint),
                                    "publisher_hash_available": False})
            elif digest(checkpoint) != json.loads(receipt.read_text())["sha256"]:
                raise ValueError("SPRC checkpoint changed")
    files = {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()
             and not p.name.endswith((".part", ".tmp", ".download.json")) and p.name != "manifest.json"
             and not p.name.startswith("pretrained_models.tar.gz")}
    write_json(output / "manifest.json", {"config": config, "files": files, "weights_downloaded": not args.sources_only})
    print(f"Prepared {args.model}: {output}", flush=True)


if __name__ == "__main__":
    main()
