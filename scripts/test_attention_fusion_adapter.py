"""Static and tensor-shape smoke tests for the UniFashion attention adapter."""
import ast
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attention_fusion import AttentionGatedFusion
from src.unifashion_runtime import _inject_attention_fusion


def test_gate_shape_and_gradients():
    fusion = AttentionGatedFusion(embedding_dim=256, hidden_dim=256)
    image_features = torch.randn(4, 256, requires_grad=True)
    text_features = torch.randn(4, 256, requires_grad=True)
    query = fusion(image_features, text_features)
    assert query.shape == (4, 256), query.shape
    assert torch.allclose(query.norm(dim=-1), torch.ones(4), atol=1e-5)
    query.square().sum().backward()
    assert all(p.grad is not None for p in fusion.parameters())
    assert all(torch.isfinite(p.grad).all() for p in fusion.parameters())
    print("GATE_256_FORWARD_BACKWARD=PASS")


def test_upstream_ast_injection():
    upstream_source = (
    ROOT.parent
    / "UniFashion"
    / "src"
    / "lavis"
    / "models"
    / "blip2_models"
    / "blip2_qformer_cir_rerank.py"
)
    if not upstream_source.is_file():
        raise FileNotFoundError(
            f"Pinned upstream source not found: {upstream_source}. "
            "Run this smoke test from the project root with the pinned repo present."
        )
    tree = ast.parse(upstream_source.read_text(encoding="utf-8"), filename=str(upstream_source))
    # Mirror the runtime import adapter's unrelated optional-import filtering.
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("rq_vae_transformer.")
    )]
    patched = _inject_attention_fusion(tree)
    compile(ast.fix_missing_locations(patched), str(upstream_source), "exec")
    class_node = next(n for n in patched.body if isinstance(n, ast.ClassDef)
                      and n.name == "Blip2QformerCirRerank")
    methods = {n.name: n for n in class_node.body if isinstance(n, ast.FunctionDef)}
    count = {
        method: sum(
            1 for node in ast.walk(method_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == "attention_fusion"
        )
        for method, method_node in methods.items()
        if method in ("forward", "inference")
    }
    assert count == {"forward": 1, "inference": 1}, count
    print("UPSTREAM_TRAIN_AND_INFERENCE_INJECTION=PASS")


if __name__ == "__main__":
    test_gate_shape_and_gradients()
    test_upstream_ast_injection()
    print("ATTENTION_ADAPTER_SMOKE_TEST=PASS")
