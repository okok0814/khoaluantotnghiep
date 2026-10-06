"""Deterministic batching and atomic checkpoint I/O for longer CIR experiments."""
from collections import defaultdict
import heapq
from pathlib import Path
import random

import torch


def make_batches(targets, batch_size, seed, epoch):
    """Keep each query, but never treat another copy of its target as a negative.

    Largest target groups are consumed first, with seeded random tie breaking.
    As in upstream drop_last=True, a short tail is reported and not trained.
    """
    if batch_size < 2:
        raise ValueError("CIR requires batch_size >= 2")
    rng = random.Random(seed + epoch)
    groups = defaultdict(list)
    for index, target in enumerate(targets):
        groups[str(target)].append(index)
    for indices in groups.values():
        rng.shuffle(indices)
    heap = [(-len(indices), rng.random(), target) for target, indices in groups.items()]
    heapq.heapify(heap)
    batches, dropped = [], []
    while heap:
        chosen = [heapq.heappop(heap) for _ in range(min(batch_size, len(heap)))]
        batch = [groups[target].pop() for _, _, target in chosen]
        if len(batch) == batch_size:
            rng.shuffle(batch)
            batches.append(batch)
        else:
            dropped.extend(batch)
        for _, _, target in chosen:
            if groups[target]:
                heapq.heappush(heap, (-len(groups[target]), rng.random(), target))
    rng.shuffle(batches)
    return batches, dropped


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_trainable_state(model, state):
    expected = {key for key in model.state_dict() if not key.startswith("visual_encoder.")}
    if set(state) != expected:
        raise ValueError("Resume checkpoint has missing/unexpected nonvision model keys")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not key.startswith("visual_encoder.") for key in result.missing_keys):
        raise ValueError(f"Resume mismatch: {result}")


def model_state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()
            if not key.startswith("visual_encoder.")}
