import copy
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.fashioniq_evaluation import digest, protocol, rank_batch, save_result
from src.fashioniq_completion import candidate, choose_original, copy_checkpoint, read_result
from scripts.evaluate_unifashion_final import restore_source_line_endings


@pytest.mark.parametrize('current,recorded,changed', [
    (b'x = 1\r\n', b'x = 1\n', True),
    (b'x = 1\n', b'x = 1\r\n', True),
    (b'x = 1\n', b'x = 1\n', False),
])
def test_upstream_line_endings_restore_exact_recorded_hash(tmp_path, current, recorded, changed):
    path = tmp_path / 'source.py'
    path.write_bytes(current)
    assert restore_source_line_endings(path, hashlib.sha256(recorded).hexdigest()) is changed
    assert path.read_bytes() == recorded


def test_upstream_code_change_is_rejected_without_rewriting(tmp_path):
    path = tmp_path / 'source.py'
    content = b'x = 2\r\n'
    path.write_bytes(content)
    with pytest.raises(ValueError, match='not just LF/CRLF'):
        restore_source_line_endings(path, hashlib.sha256(b'x = 1\n').hexdigest())
    assert path.read_bytes() == content


def fixture_result(tmp_path):
    data, result = tmp_path / 'data', tmp_path / 'result'
    data.mkdir()
    gallery = [f'i{x}' for x in range(61)]
    queries = pd.DataFrame([{'category': c, 'candidate': 'i0', 'target': 'i1', 'text': 'red'}
                            for c in ('dress', 'shirt', 'toptee')])
    queries.to_csv(data / 'queries.csv', index=False)
    for c in queries.category:
        (data / f'split.{c}.val.json').write_text(json.dumps(gallery))
    (data / 'gallery_captions.json').write_text('{}')
    q, galleries, fp = protocol(data)
    records = []
    for c in q.category:
        records += rank_batch(np.arange(61, 0, -1)[None], q[q.category == c], gallery)
    p = {'checkpoint_sha256': 'example', 'optimizer_steps': 60000, 'evaluation_scope': 'full_validation',
         'training_fingerprint': {'code': {'unifashion_runtime.py': 'original-runtime'}},
         'code_sha256': {'unifashion_runtime.py': 'original-runtime'},
         'gallery_captions_sha256': digest(data / 'gallery_captions.json'), 'training_config': {'seed': 42}}
    save_result(result, 'unifashion_image_caption_score', records, galleries, fp, p)
    (result / 'status.json').write_text(json.dumps({'status': 'evaluation_completed', 'queries': 3,
                                                  'checkpoint_sha256': 'example'}))
    return data, result


def test_validates_saved_ranks_and_rejects_reference_and_incomplete_coverage(tmp_path):
    data, folder = fixture_result(tmp_path)
    read_result(folder, 'unifashion_image_caption_score', data, 'original', 60000)
    csv = folder / 'unifashion_image_caption_score_per_query.csv'
    frame = pd.read_csv(csv)
    changed = frame.copy()
    top = json.loads(changed.loc[0, 'top50'])
    top[-1] = 'i0'
    changed.loc[0, 'top50'] = json.dumps(top)
    changed.to_csv(csv, index=False)
    with pytest.raises(ValueError, match='Reference'):
        read_result(folder, 'unifashion_image_caption_score', data, 'original', 60000)
    frame.iloc[:2].to_csv(csv, index=False)
    with pytest.raises(ValueError, match='Incomplete'):
        read_result(folder, 'unifashion_image_caption_score', data, 'original', 60000)


def test_refuses_model_variant_runtime_and_screening_mismatches(tmp_path):
    data, folder = fixture_result(tmp_path)
    with pytest.raises(ValueError, match='variant'):
        read_result(folder, 'unifashion_image_caption_score', data, 'attention')
    path = folder / 'unifashion_image_caption_score.json'
    r = json.loads(path.read_text())
    r['provenance']['code_sha256']['unifashion_runtime.py'] = 'gate-runtime'
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='runtime'):
        read_result(folder, 'unifashion_image_caption_score', data, 'original')
    r['provenance']['evaluation_scope'] = 'screening_only_reduced_gallery'
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='Screening'):
        read_result(folder, 'unifashion_image_caption_score', data, 'original')


def test_selection_retains_base_on_ties_and_requires_same_run(tmp_path):
    data, folder = fixture_result(tmp_path)
    a = candidate('step60000', read_result(folder, 'unifashion_image_caption_score', data, 'original'))
    b = copy.deepcopy(a)
    b['name'] = 'step90000'
    assert choose_original(a, b)['name'] == 'step60000'
    b['mean_R'] += 0.1
    assert choose_original(a, b)['name'] == 'step90000'
    b['training_config']['seed'] = 7
    with pytest.raises(ValueError, match='configs'):
        choose_original(a, b)


def test_snapshot_copy_preserves_source_and_refuses_overwrite_or_wrong_hash(tmp_path):
    source, destination = tmp_path / 'last.pt', tmp_path / 'snapshot.pt'
    expected = {'optimizer_steps': 2, 'training_fingerprint': {'run': 'original'}, 'training_config': {'seed': 42}}
    torch.save({'model': {'weight': torch.ones(2)}, 'history': [{'step': 1}, {'step': 2}],
                'fingerprint': expected['training_fingerprint'], 'config': expected['training_config'],
                'epochs': [{}], 'cursor': [1, 0]}, source)
    before = digest(source)
    receipt = copy_checkpoint(source, destination, expected)
    assert receipt['copy_sha256_verified'] and receipt['cpu_deserialization_verified']
    assert digest(source) == digest(destination) == before
    assert copy_checkpoint(source, destination, expected)['checkpoint_sha256'] == before
    with pytest.raises(ValueError, match='Source differs'):
        copy_checkpoint(source, tmp_path / 'wrong.pt', {**expected, 'checkpoint_sha256': 'wrong'})
    destination.write_bytes(b'existing different checkpoint')
    with pytest.raises(ValueError, match='Existing destination'):
        copy_checkpoint(source, destination, expected)
    assert digest(source) == before


def test_report_keeps_pending_metrics_and_requires_matching_copy_receipt(tmp_path):
    from scripts.report_unifashion_completion import generate
    data, base = fixture_result(tmp_path)
    inputs, out = tmp_path / 'inputs', tmp_path / 'report'
    inputs.mkdir()
    shutil.copytree(data, inputs / 'evaluation_data')
    baseline = inputs / 'baselines'
    baseline.mkdir()
    r = json.loads((base / 'unifashion_image_caption_score.json').read_text())
    frame = pd.read_csv(base / 'unifashion_image_caption_score_per_query.csv')
    b = copy.deepcopy(r)
    b['method'] = 'fashionclip_additive'
    (baseline / 'fashionclip_additive.json').write_text(json.dumps(b))
    frame.to_csv(baseline / 'fashionclip_additive_per_query.csv', index=False)
    no_cap = copy.deepcopy(r)
    no_cap['method'] = 'unifashion_image_score'
    (base / 'unifashion_image_score.json').write_text(json.dumps(no_cap))
    frame.to_csv(base / 'unifashion_image_score_per_query.csv', index=False)
    shutil.copytree(base, inputs / 'base')
    verification = {'training_completed': True, 'optimizer_steps': 90000,
                    'fingerprint': r['provenance']['training_fingerprint'],
                    'best_monitor_loss': 0.2, 'best_monitor_epoch': 7, 'recorded_update_hours': 1}
    (inputs / 'training_verification.json').write_text(json.dumps(verification))
    pd.DataFrame([{'epoch': 10, 'step': 90000, 'train_loss': 0.1, 'monitor_loss': 0.2}]).to_csv(inputs / 'training_epochs.csv', index=False)
    state = generate(inputs, out)
    assert not state['last_week_task_completed'] and not state['original_final_evaluated']
    table = pd.read_csv(out / 'model_comparison.csv')
    assert not table.model.str.contains('step90000|attention').any()
    final = tmp_path / 'final'
    shutil.copytree(base, final)
    for method in ('unifashion_image_score', 'unifashion_image_caption_score'):
        path = final / (method + '.json')
        result = json.loads(path.read_text())
        result['provenance']['optimizer_steps'] = 90000
        path.write_text(json.dumps(result))
    state = generate(inputs, out, final)
    assert state['original_final_evaluated'] and not state['last_week_task_completed']
    receipt = {'checkpoint_sha256': 'wrong', 'copy_sha256_verified': True, 'cpu_deserialization_verified': True}
    (out / 'checkpoint_copy.json').write_text(json.dumps(receipt))
    assert not generate(inputs, out, final)['last_week_task_completed']
    receipt['checkpoint_sha256'] = 'example'
    (out / 'checkpoint_copy.json').write_text(json.dumps(receipt))
    state = generate(inputs, out, final)
    assert state['last_week_task_completed']
    assert not state['three_model_validation_comparison_completed']
    assert not state['original_test_task_completed']
