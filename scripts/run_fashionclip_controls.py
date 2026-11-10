"""Re-evaluate the historical gate or train equal-budget frozen-feature fusion controls."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.attention_fusion import AttentionGatedFusion, in_batch_contrastive_loss
from src.fashioniq_evaluation import digest, normalize, protocol
from src.paper_comparison import (cached_embeddings, evaluate_vectors, lock_run, write_json)


class WeightedAdditive(nn.Module):
    """Learn one global image weight on TRAIN only, with the same contrastive objective."""
    def __init__(self, dim=512):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(()))

    def forward(self, image, text):
        a = self.logit.sigmoid()
        return F.normalize(a * F.normalize(image, dim=-1) + (1-a) * F.normalize(text, dim=-1), dim=-1)


class ResidualMLP(nn.Module):
    """Two-layer composition control, within 0.2% of the gate's parameter count at D=512."""
    def __init__(self, dim=512):
        super().__init__()
        gate_count = sum(p.numel() for p in AttentionGatedFusion(dim, dim).parameters())
        hidden = round((gate_count-dim) / (3*dim+1))
        self.layers = nn.Sequential(nn.Linear(2*dim, hidden), nn.ReLU(), nn.Dropout(.1),
                                    nn.Linear(hidden, dim))

    def forward(self, image, text):
        image, text = F.normalize(image, dim=-1), F.normalize(text, dim=-1)
        return F.normalize((image+text)/2 + self.layers(torch.cat([image, text], -1)), dim=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings", type=Path, default=ROOT / "logs")
    p.add_argument("--eval-data", type=Path, default=ROOT / "logs/week41_fashioniq_evaluation/evaluation_data")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--legacy-checkpoint", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.learning_rate <= 0:
        raise ValueError("Invalid training configuration")
    torch.set_num_threads(2)
    queries, galleries, fingerprint = protocol(args.eval_data)
    images, texts, metadata, lookup, rows, sources = cached_embeddings(args.embeddings, queries)
    sources["code_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in
        [Path(__file__), ROOT / "src/attention_fusion.py", ROOT / "src/paper_comparison.py", ROOT / "src/fashioniq_evaluation.py"]}
    val_im = normalize(images[[lookup[x] for x in queries.candidate]])
    val_tx = normalize(texts[rows.embedding_row.astype(int)])
    args.output.mkdir(parents=True, exist_ok=True)

    def evaluate(model, name, folder, provenance):
        model.eval()
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(queries), 256):
                vectors.append(model(torch.from_numpy(val_im[start:start+256]).to(args.device),
                                     torch.from_numpy(val_tx[start:start+256]).to(args.device)).cpu().numpy())
        result = evaluate_vectors(name, queries, galleries, fingerprint, images, lookup,
                                 np.concatenate(vectors), folder, provenance)
        print(name, result["summary"][-2], flush=True)

    if args.legacy_checkpoint:
        spec = {"protocol": fingerprint, "sources": sources, "checkpoint_sha256": digest(args.legacy_checkpoint),
                "model": "historical standalone FashionCLIP gate; not integrated UniFashion", "device": args.device}
        out = args.output / "legacy_gate"
        lock_run(out, spec)
        model = AttentionGatedFusion(images.shape[1], images.shape[1]).to(args.device)
        model.load_state_dict(torch.load(args.legacy_checkpoint, map_location="cpu", weights_only=True), strict=True)
        evaluate(model, "fashionclip_legacy_gate", out, spec)
        return

    # Match all official category TRAIN triplets; exclude duplicated 'all' exports, not samples.
    if sources["features"].get("pixels") != "same audited resized files as UniFashion":
        raise ValueError("Extract matched image features first with extract_paper_fashionclip.py")
    train = metadata[metadata.split == "train"].reset_index(drop=True)
    if len(train) != 18000 or train.category.value_counts().to_dict() != {"dress": 5985, "shirt": 5988, "toptee": 6027}:
        raise ValueError("Expected all 18,000 official category training triplets")
    tensors = [torch.from_numpy(normalize(x)) for x in
        (images[[lookup[x] for x in train.candidate]], texts[train.embedding_row.astype(int)],
         images[[lookup[x] for x in train.target]])]
    for name, vectors in (("fashionclip_image_only_matched", val_im), ("fashionclip_text_only_matched", val_tx),
                          ("fashionclip_additive_matched", normalize(val_im+val_tx))):
        # The suffix describes the current feature bundle; legacy caches are marked explicitly in provenance.
        evaluate_vectors(name, queries, galleries, fingerprint, images, lookup, vectors,
                         args.output / "fixed", sources)
    for seed in args.seeds:
        for name, cls in (("weighted", WeightedAdditive), ("mlp", ResidualMLP), ("gate", AttentionGatedFusion)):
            torch.manual_seed(seed)
            model = cls(images.shape[1]).to(args.device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
            out = args.output / f"{name}_seed{seed}"
            spec = {"protocol": fingerprint, "sources": sources, "seed": seed, "epochs": args.epochs,
                    "batch_size": args.batch_size, "lr": args.learning_rate, "weight_decay": 1e-4,
                    "temperature": .07, "drop_last": True, "model": name, "device": args.device,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "selection": "fixed final epoch; no validation-based selection",
                    "loss": "one-way in-batch CE, identical to historical gate; duplicate targets not masked"}
            lock_run(out, spec)
            checkpoint = out / "resume.pt"
            start_epoch, history = 0, []
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location=args.device, weights_only=True)
                model.load_state_dict(saved["model"])
                opt.load_state_dict(saved["optimizer"])
                start_epoch, history = saved["epoch"], saved["history"]
                torch.set_rng_state(saved["cpu_rng"].cpu())
                if args.device.startswith("cuda"):
                    torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
            for epoch in range(start_epoch, args.epochs):
                model.train()
                order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed+epoch))
                losses, start = [], time.monotonic()
                for offset in range(0, len(train)-args.batch_size+1, args.batch_size):
                    idx = order[offset:offset+args.batch_size]
                    im, tx, target = (x[idx].to(args.device) for x in tensors)
                    opt.zero_grad(set_to_none=True)
                    loss, _ = in_batch_contrastive_loss(model(im, tx), target, .07)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
                    opt.step()
                    losses.append(float(loss.detach()))
                history.append({"epoch": epoch+1, "loss": float(np.mean(losses)), "updates": len(losses),
                                "seconds": time.monotonic()-start})
                print(name, seed, history[-1], flush=True)
                saved = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch+1,
                         "history": history, "cpu_rng": torch.get_rng_state(),
                         "cuda_rng": torch.cuda.get_rng_state() if args.device.startswith("cuda") else None}
                torch.save(saved, out / "resume.pt.tmp")
                (out / "resume.pt.tmp").replace(checkpoint)
                write_json(out / "training.json", {"specification": spec, "history": history})
            provenance = {**spec, "checkpoint_sha256": digest(checkpoint)}
            if name == "weighted":
                provenance["image_weight"] = float(model.logit.sigmoid().detach())
            evaluate(model, f"fashionclip_{name}_seed{seed}", out, provenance)


if __name__ == "__main__":
    main()
