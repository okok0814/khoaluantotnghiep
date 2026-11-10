"""Resume external-model and pre-fine-tuning evaluations under the project's common protocol."""
import argparse
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import torch
from PIL import Image
from src.fashioniq_evaluation import CATEGORIES, digest, protocol, rank_batch, save_result
from src.paper_comparison import lock_run, write_json


def lock_evaluation_run(output, spec):
    """Preserve an initialization-only attempt when code is fixed before any inference.

    A cache, provenance file, result, or any unknown file prevents automatic
    archival. Runs that have started still use the strict resume identity check.
    """
    output = Path(output)
    previous = output / "run_spec.json"
    if previous.is_file() and {p.name for p in output.iterdir()} == {"run_spec.json"}:
        old = json.loads(previous.read_text(encoding="utf-8"))
        if old != spec and old.get("model") == spec.get("model"):
            archived = output.with_name(output.name + "_unstarted_" + uuid.uuid4().hex)
            if archived.resolve().parent != output.resolve().parent:
                raise ValueError("Archive must stay beside the original output directory")
            output.rename(archived)
            print("Preserved initialization-only attempt:", archived, flush=True)
    lock_run(output, spec)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["clip4cir", "sprc", "unifashion-initial"], required=True)
    for arg in ("assets", "data-dir", "images-dir", "eval-data", "output", "original-code", "unifashion-assets", "upstream", "reference-result"):
        p.add_argument("--"+arg, type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    if not torch.cuda.is_available() or args.batch_size < 1:
        raise RuntimeError("Use a Colab GPU and a positive batch size")
    torch.set_num_threads(2)
    queries, galleries, fingerprint = protocol(args.eval_data)
    if len(queries) != 6016:
        raise ValueError("This is a full-validation evaluator; screening results are not allowed")
    captions = json.loads((args.eval_data / "gallery_captions.json").read_text())
    manifest_path = args.data_dir / "images_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads((args.data_dir / "data_audit.json").read_text())
    reference = json.loads(args.reference_result.read_text())["provenance"]
    if digest(args.data_dir / "data_audit.json") != reference["training_fingerprint"]["data_audit"]:
        raise ValueError("Use the same audited data bundle as the final UniFashion evaluation")
    if digest(manifest_path) != audit["files"]["images_manifest.json"]:
        raise ValueError("Audited image manifest changed")
    for name in sorted(set().union(*map(set, galleries.values()))):
        if digest(args.images_dir / manifest[name]["file"]) != manifest[name]["sha256"]:
            raise ValueError("Image bytes changed: " + name)
    from src.paper_model_adapters import Clip4Cir, Sprc, UniFashionInitial
    model_cls = {"clip4cir": Clip4Cir, "sprc": Sprc, "unifashion-initial": UniFashionInitial}[args.model]
    asset_manifest = None
    if args.model != "unifashion-initial":
        asset_manifest = json.loads((args.assets / args.model / "manifest.json").read_text())
        if not asset_manifest["weights_downloaded"]:
            raise ValueError("Model weights have not been downloaded")
        config = json.loads((ROOT / "configs/paper_comparisons.json").read_text())[args.model]
        if asset_manifest["config"] != config:
            raise ValueError("Model source/config differs from the pinned comparison")
        for name, sha in asset_manifest["files"].items():
            if digest(args.assets / args.model / name) != sha:
                raise ValueError("Pinned source/weight changed: " + name)
    spec = {"model": args.model, "protocol": fingerprint, "image_manifest_sha256": digest(manifest_path),
            "caption_sha256": digest(args.eval_data / "gallery_captions.json"),
            "reference_sha256": digest(args.reference_result), "original_code_sha256": digest(args.original_code),
            "asset_manifest": asset_manifest, "precision": "fp32", "batch_size": args.batch_size,
            "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
            "code_sha256": {str(f.relative_to(ROOT)): digest(f) for f in [Path(__file__),
                ROOT / "src/paper_model_adapters.py", ROOT / "src/paper_comparison.py", ROOT / "src/fashioniq_evaluation.py",
                ROOT / "src/unifashion_sanity_data.py"]}}
    lock_evaluation_run(args.output, spec)
    adapter = model_cls(**vars(args), scratch=ROOT / ".cache/paper_runtime")
    provenance = {**spec, **adapter.provenance, "evaluation_scope": "full_validation",
                  "no_reranking": True, "caption_resource": "released next_llava.val captions; only UniFashion uses them"}
    write_json(args.output / "provenance.json", provenance)

    def images(names):
        values = []
        for name in names:
            with Image.open(args.images_dir / manifest[name]["file"]) as image:
                values.append(adapter.transform(image.convert("RGB")))
        return torch.stack(values).cuda()

    def valid_cache(path):
        receipt = path.with_suffix(path.suffix+".json")
        return path.exists() and receipt.exists() and digest(path) == json.loads(receipt.read_text())["sha256"]

    def seal(path):
        write_json(path.with_suffix(path.suffix+".json"), {"sha256": digest(path)})

    predictions = {method: [] for method in adapter.methods}
    start_time = time.monotonic()
    with torch.inference_mode():
        for category in CATEGORIES:
            gallery = galleries[category]
            rows = queries[queries.category == category].reset_index(drop=True)
            cache = args.output / "cache" / category
            cache.mkdir(parents=True, exist_ok=True)
            features = {}
            for start in range(0, len(gallery), 128):
                path = cache / f"gallery_{start:06d}.pt"
                if valid_cache(path):
                    chunk = torch.load(path, map_location="cpu", weights_only=True)
                else:
                    chunk = {}
                    for offset in range(start, min(start+128, len(gallery)), args.batch_size):
                        ids = gallery[offset:min(offset+args.batch_size,start+128)]
                        result = adapter.gallery(images(ids), [captions[category][x] for x in ids])
                        for key, value in result.items():
                            if not torch.isfinite(value).all():
                                raise ValueError("Nonfinite gallery features")
                            chunk.setdefault(key, []).append(value.float().cpu())
                    chunk = {k: torch.cat(v) for k,v in chunk.items()}
                    torch.save(chunk, path.with_suffix(".tmp"))
                    path.with_suffix(".tmp").replace(path)
                    seal(path)
                for key, value in chunk.items():
                    features.setdefault(key, []).append(value)
                print(f"{args.model} {category}: gallery {min(start+128,len(gallery))}/{len(gallery)}", flush=True)
            targets = {k: torch.cat(v).cuda() for k,v in features.items()}
            del features
            for start in range(0, len(rows), 64):
                path = cache / f"queries_{start:06d}.json"
                if valid_cache(path):
                    records = json.loads(path.read_text())
                else:
                    records = {m: [] for m in adapter.methods}
                    for offset in range(start, min(start+64,len(rows)), args.batch_size):
                        batch = rows.iloc[offset:min(offset+args.batch_size,start+64)]
                        scores = adapter.scores(images(batch.candidate), batch.text.tolist(), targets)
                        for method, matrix in scores.items():
                            records[method] += rank_batch(matrix.float().cpu().numpy(), batch, gallery)
                    write_json(path, records)
                    seal(path)
                for method, result in records.items():
                    predictions[method].extend(result)
                print(f"{args.model} {category}: queries {min(start+64,len(rows))}/{len(rows)}", flush=True)
            del targets
    provenance["this_invocation_seconds"] = time.monotonic()-start_time
    for method, records in predictions.items():
        if len(records) != len(queries):
            raise ValueError("Incomplete prediction coverage")
        result = save_result(args.output, method, records, galleries, fingerprint, provenance)
        from src.fashioniq_completion import read_result
        read_result(args.output, method, args.eval_data)
        print(method, result["summary"][-2], flush=True)
    write_json(args.output / "status.json", {"status": "evaluation_completed", "queries": len(queries), "methods": adapter.methods})


if __name__ == "__main__":
    main()
