"""Comparison protocol and small official-model tests; fixtures are not thesis scores."""
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import numpy as np
import pandas as pd
import pytest
import torch
from src.paper_comparison import cached_embeddings, lock_run, paired_bootstrap
from scripts.prepare_paper_models import PartReader, extract_selected
from scripts.run_fashionclip_controls import WeightedAdditive, ResidualMLP
from src.attention_fusion import AttentionGatedFusion

ROOT = Path(__file__).resolve().parents[1]


def test_paired_bootstrap_is_paired_order_independent_and_macro():
    rows = [{"category": c, "candidate": str(i), "target": str(i+10), "hit@10": h, "hit@50": 1}
            for c, n, h in (("dress", 2, 1), ("shirt", 4, 0), ("toptee", 8, 0)) for i in range(n)]
    a = pd.DataFrame(rows)
    b = a.sample(frac=1, random_state=7)
    for row in paired_bootstrap(a, b, repeats=200):
        assert row['estimate'] == row['lower95'] == row['upper95'] == 0
    estimate = paired_bootstrap(a, repeats=200)
    assert estimate[0]['estimate'] == pytest.approx(100/3)
    assert estimate[1]['estimate'] == pytest.approx(100)
    b.loc[b.index[0], 'target'] = 'different'
    with pytest.raises(ValueError, match='identical query'):
        paired_bootstrap(a, b, repeats=200)


def test_resume_rejects_changed_inputs_and_unknown_existing_files(tmp_path):
    path = tmp_path / 'run'
    lock_run(path, {'model': 'a', 'hash': 'one'})
    lock_run(path, {'model': 'a', 'hash': 'one'})
    with pytest.raises(ValueError, match='changed'):
        lock_run(path, {'model': 'a', 'hash': 'two'})
    orphan = tmp_path / 'orphan'
    orphan.mkdir()
    (orphan / 'old.pt').touch()
    with pytest.raises(ValueError, match='no run specification'):
        lock_run(orphan, {})


def test_metadata_mapping_keeps_original_rows_excludes_test_and_all(tmp_path):
    pd.DataFrame({'image_id': ['a', 'b']}).to_csv(tmp_path/'fashionclip_image_ids.csv', index=False)
    np.save(tmp_path/'fashionclip_image_embeddings.npy', np.eye(2, dtype=np.float32))
    rows = [dict(category=c, split=s, candidate='a', target=t, text=text)
            for c,s,t,text in [('all','val','b','wrong'), ('dress','test','','one'),
                               ('dress','test','','two'), ('dress','val','b','right')]]
    pd.DataFrame(rows).to_csv(tmp_path/'fashionclip_text_metadata.csv', index=False)
    np.save(tmp_path/'fashionclip_text_embeddings.npy', np.ones((4,2), dtype=np.float32))
    queries = pd.DataFrame([rows[-1]]).drop(columns='split')
    _,_,_,_,mapped,_ = cached_embeddings(tmp_path, queries)
    assert mapped.embedding_row.tolist() == [3]
    queries.text = 'wrong'
    with pytest.raises(ValueError, match='differs'):
        cached_embeddings(tmp_path, queries)


def test_matched_control_capacity_and_trainable_text_path():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    models = [WeightedAdditive(), ResidualMLP(), AttentionGatedFusion()]
    counts = [sum(p.numel() for p in m.parameters()) for m in models]
    assert counts[0] == 1
    assert abs(counts[1]-counts[2])/counts[2] < .002
    image, text = torch.randn(3,512), torch.randn(3,512)
    for model in models:
        model.eval()
        out = model(image, text)
        torch.testing.assert_close(out.norm(dim=-1), torch.ones(3))
        assert not torch.allclose(out, model(image, torch.zeros_like(text)))
        (-out[:,0].sum()).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def archive_bytes(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as t:
        for name, content in entries:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            t.addfile(info, io.BytesIO(content))
    return stream.getvalue()


def test_split_archive_selects_variant_and_blocks_traversal(tmp_path):
    raw = archive_bytes([('CIRR/RN50x4_fullft/ignore.pt', b'no'),
                         ('FashionIQ/RN50x4_fullft/fiq_clip_best.pt', b'yes')])
    paths = []
    for i in range(0,len(raw),19):
        p = tmp_path / f'part-{i}'
        p.write_bytes(raw[i:i+19])
        paths.append(p)
    reader = PartReader(paths)
    extract_selected(reader, tmp_path/'out')
    reader.close()
    assert (tmp_path/'out/FashionIQ/RN50x4_fullft/fiq_clip_best.pt').read_bytes() == b'yes'
    assert not (tmp_path/'out/CIRR').exists()
    with pytest.raises(ValueError, match='Unsafe'):
        extract_selected(io.BytesIO(archive_bytes([('../escape',b'x')])), tmp_path/'out')


def test_original_runtime_rejects_wrong_variant(tmp_path):
    import zipfile
    from src.paper_model_adapters import original_runtime
    bad = tmp_path/'bad.zip'
    with zipfile.ZipFile(bad,'w') as z:
        z.writestr('src/unifashion_runtime.py','injected fusion')
    with pytest.raises(ValueError, match='fingerprint'):
        original_runtime(bad, tmp_path/'runtime')


class SprcLoadFixture(torch.nn.Module):
    """Tiny retrieval tree with the exact 30522/30523 unused-bias mismatch."""
    def __init__(self):
        super().__init__()
        self.Qformer = torch.nn.Module()
        self.Qformer.bert = torch.nn.Module()
        embeddings = torch.nn.Module()
        embeddings.word_embeddings = torch.nn.Embedding(30523, 4)
        embeddings.position_embeddings = torch.nn.Embedding(16, 4)
        embeddings.register_buffer('position_ids', torch.arange(16).expand((1, -1)))
        self.Qformer.bert.embeddings = embeddings
        self.Qformer.cls = torch.nn.Module()
        self.Qformer.cls.predictions = torch.nn.Module()
        self.Qformer.cls.predictions.bias = torch.nn.Parameter(torch.zeros(30522))
        self.visual_encoder = torch.nn.Linear(3, 4)
        self.text_proj = torch.nn.Linear(4, 2)


@pytest.mark.parametrize('omit_positions', [False, True])
def test_sprc_load_ignores_only_unused_head_and_materializes_meta(omit_positions):
    from src.paper_model_adapters import load_sprc_retrieval_weights
    checkpoint = SprcLoadFixture().state_dict()
    checkpoint['Qformer.cls.predictions.bias'] = torch.zeros(30523)
    with torch.device('meta'):
        broken = SprcLoadFixture()
        model = SprcLoadFixture()
    # strict=False still fails on the reported shape mismatch.
    with pytest.raises(RuntimeError, match='Qformer.cls.predictions.bias'):
        broken.load_state_dict(checkpoint, strict=False, assign=True)
    vision = {k.removeprefix('visual_encoder.'): v for k,v in checkpoint.items() if k.startswith('visual_encoder.')}
    fine = {k:v for k,v in checkpoint.items() if not k.startswith('visual_encoder.')}
    position_key = 'Qformer.bert.embeddings.position_ids'
    if omit_positions:
        fine.pop(position_key)
    report = load_sprc_retrieval_weights(model, vision, {}, fine)
    assert report['ignored_language_head_keys'] == ['Qformer.cls.predictions.bias']
    assert report['reconstructed_buffers'] == ([position_key] if omit_positions else [])
    assert report['missing_retrieval_keys'] == []
    assert isinstance(model.Qformer.cls, torch.nn.Identity)
    for key, tensor in model.state_dict().items():
        assert not tensor.is_meta
        torch.testing.assert_close(tensor, checkpoint[key])


@pytest.mark.parametrize('problem', ['missing', 'shape', 'unexpected'])
def test_sprc_load_rejects_bad_retrieval_weights(problem):
    from src.paper_model_adapters import load_sprc_retrieval_weights
    model = SprcLoadFixture()
    state = model.state_dict()
    if problem == 'missing':
        state.pop('text_proj.weight')
        message = 'active tensors missing'
    elif problem == 'shape':
        # The real token embeddings remain strict even when LM-head sizes differ.
        state['Qformer.bert.embeddings.word_embeddings.weight'] = torch.zeros(30522, 4)
        message = 'retrieval weight shape mismatch'
    else:
        state['unknown.weight'] = torch.ones(1)
        message = 'Unexpected SPRC weights'
    with pytest.raises(ValueError, match=message):
        load_sprc_retrieval_weights(model, {}, {}, state)


def test_evaluation_restart_preserves_only_unstarted_attempt(tmp_path):
    from scripts.evaluate_paper_models import lock_evaluation_run
    out = tmp_path / 'sprc'
    old = {'model': 'sprc', 'code_sha256': 'before'}
    new = {'model': 'sprc', 'code_sha256': 'fixed'}
    lock_run(out, old)
    original = (out / 'run_spec.json').read_bytes()
    lock_evaluation_run(out, new)
    archived = list(tmp_path.glob('sprc_unstarted_*'))
    assert len(archived) == 1
    assert (archived[0] / 'run_spec.json').read_bytes() == original
    lock_evaluation_run(out, new)
    assert len(list(tmp_path.glob('sprc_unstarted_*'))) == 1


@pytest.mark.parametrize('existing', ['provenance.json', 'status.json', 'cache'])
def test_evaluation_restart_never_archives_started_runs(tmp_path, existing):
    from scripts.evaluate_paper_models import lock_evaluation_run
    out = tmp_path / 'sprc'
    lock_run(out, {'model': 'sprc', 'code_sha256': 'before'})
    if existing == 'cache':
        (out / existing).mkdir()
    else:
        (out / existing).write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        lock_evaluation_run(out, {'model': 'sprc', 'code_sha256': 'fixed'})
    assert (out / existing).exists()
    assert not list(tmp_path.glob('sprc_unstarted_*'))


def test_official_combiner_tiny_inference():
    from src.paper_model_adapters import module_file
    source = ROOT/'.cache/paper_assets/clip4cir/source/src/combiner.py'
    if not source.exists():
        pytest.skip('Run prepare_paper_models.py --model clip4cir --sources-only')
    model = module_file('test_official_combiner', source).Combiner(8,16,32).eval()
    a,b = torch.randn(3,8), torch.randn(3,8)
    whole = model.combine_features(a,b)
    separate = torch.cat([model.combine_features(a[i:i+1],b[i:i+1]) for i in range(3)])
    torch.testing.assert_close(whole, separate)
    torch.testing.assert_close(whole.norm(dim=-1), torch.ones(3))


def test_official_sprc_tiny_query_batch_one(tmp_path):
    source = ROOT/'.cache/paper_assets/sprc/source/src/lavis'
    if not source.exists():
        pytest.skip('Run prepare_paper_models.py --model sprc --sources-only')
    command = '''
import sys
from pathlib import Path
import torch
from torch import nn
from src.paper_model_adapters import original_runtime, sprc_class, load_sprc_retrieval_weights
from accelerate import init_empty_weights
root, scratch = Path(sys.argv[1]), Path(sys.argv[2])
helper = original_runtime(root/'checkpoints/unifashion_week8_code.zip',scratch)
official = sprc_class(root/'.cache/paper_assets/sprc/source/src/lavis',helper,scratch)
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from transformers import BertTokenizer
torch.set_num_threads(2)
class TinyVision(nn.Module):
    num_features=16
    def __init__(self):
        super().__init__(); self.proj=nn.Conv2d(3,16,4,stride=4)
    def forward(self,x):
        return self.proj(x).flatten(2).transpose(1,2)
class Tiny(official):
    @classmethod
    def init_tokenizer(cls):
        t=BertTokenizer.from_pretrained(str(root/'checkpoints/unifashion/bert-base-uncased'),local_files_only=True)
        t.add_special_tokens({'bos_token':'[DEC]'})
        return t
    def init_vision_encoder(self,*a):
        return TinyVision(),nn.LayerNorm(16)
    @classmethod
    def init_Qformer(cls,n,width,freq):
        cfg=BertConfig(vocab_size=30523,hidden_size=32,num_hidden_layers=2,num_attention_heads=4,intermediate_size=64,encoder_width=width,add_cross_attention=True,cross_attention_freq=2,query_length=n)
        return BertLMHeadModel(cfg),nn.Parameter(torch.randn(1,n,32)*.02)
m=Tiny(embed_dim=16,max_txt_len=12).eval()
with init_empty_weights():
    restored=Tiny(embed_dim=16,max_txt_len=12)
restored.Qformer.cls.predictions.bias=nn.Parameter(torch.empty(30522,device='meta'))
state=m.state_dict()
state.pop('Qformer.bert.embeddings.position_ids',None)
report=load_sprc_retrieval_weights(restored,{}, {},state)
restored=restored.float().cpu().eval()
assert report['missing_retrieval_keys']==[]
with torch.inference_mode():
    target_images=torch.randn(7,3,16,16)
    ref_images=torch.randn(2,3,16,16)
    features,_=m.extract_target_features(target_images)
    ref=m.ln_vision(m.visual_encoder(ref_images))
    texts=['make it red','longer sleeves']
    whole=m.inference(ref,features,texts).reshape(2,7)
    separate=torch.cat([m.inference(ref[i:i+1],features,texts[i:i+1]).reshape(1,7) for i in range(2)])
    assert torch.isfinite(whole).all()
    torch.testing.assert_close(whole,separate,atol=1e-5,rtol=1e-5)
    loaded_features,_=restored.extract_target_features(target_images)
    loaded_ref=restored.ln_vision(restored.visual_encoder(ref_images))
    loaded_scores=restored.inference(loaded_ref,loaded_features,texts).reshape(2,7)
    torch.testing.assert_close(whole,loaded_scores,atol=1e-5,rtol=1e-5)
print('SPRC official tiny inference passed')
'''
    done = subprocess.run([sys.executable,'-B','-c',command,str(ROOT),str(tmp_path)], capture_output=True,text=True)
    assert done.returncode == 0, done.stdout + done.stderr
