"""Evidence validation and checkpoint selection shared by the completion workflow."""
import json
from pathlib import Path
import shutil
import uuid
import numpy as np
import pandas as pd
from src.fashioniq_evaluation import digest, protocol, summarize


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_result(folder, method, eval_data, variant=None, expected_steps=None):
    folder = Path(folder)
    result = json.loads((folder / (method + '.json')).read_text(encoding='utf-8'))
    queries, galleries, fingerprint = protocol(eval_data)
    require(result['method'] == method and result['protocol'] == fingerprint, 'Different method/protocol')
    frame = pd.read_csv(folder / (method + '_per_query.csv'), keep_default_na=False)
    keys = ['category', 'candidate', 'target']
    require(len(frame) == len(queries) and not frame.duplicated(keys).any(), 'Incomplete/duplicate predictions')
    expected = queries.set_index(keys).sort_index()
    actual = frame.set_index(keys).sort_index()
    pd.testing.assert_frame_equal(actual[expected.columns].astype(str), expected.astype(str))
    sets = {k: set(v) for k, v in galleries.items()}
    for row in frame.to_dict('records'):
        rank, top = row['target_rank'], json.loads(row['top50'])
        require(isinstance(rank, int) and 1 <= rank < len(galleries[row['category']]), 'Invalid rank')
        require(len(top) == len(set(top)) == min(50, len(galleries[row['category']]) - 1)
                and set(top) <= sets[row['category']], 'Invalid top50')
        require(row['candidate'] not in top, 'Reference was ranked')
        require((row['target'] in top) == (rank <= 50), 'Target membership/rank disagree')
        if rank <= 50:
            require(top[rank-1] == row['target'], 'Target position differs')
        for k in (10, 50):
            require(row[f'hit@{k}'] == int(rank <= k), 'Hit/rank mismatch')
    calculated = summarize(frame.to_dict('records'), galleries)
    require(len(calculated) == len(result['summary']), 'Summary rows differ')
    for a, b in zip(calculated, result['summary'], strict=True):
        for key, value in a.items():
            require(np.isclose(value, b[key], atol=1e-10, rtol=0) if key.startswith('Recall')
                    else value == b[key], 'Summary mismatch: ' + key)
    if variant is not None:
        status = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
        p = result['provenance']
        require(status['status'] == 'evaluation_completed' and status['queries'] == len(queries),
                'Evaluation did not complete')
        require(status['checkpoint_sha256'] == p['checkpoint_sha256'], 'Result/checkpoint mismatch')
        require(p.get('evaluation_scope') == 'full_validation', 'Screening is not full validation')
        code = p['training_fingerprint']['code']
        observed = 'attention' if 'attention_fusion.py' in code else 'original'
        require(observed == variant, 'Wrong model variant')
        require(p['code_sha256']['unifashion_runtime.py'] == code['unifashion_runtime.py'],
                'Evaluation runtime differs from training')
        if variant == 'attention':
            require(p['code_sha256'].get('attention_fusion.py') == code['attention_fusion.py'],
                    'Fusion implementation differs from training')
        if expected_steps is not None:
            require(p['optimizer_steps'] == expected_steps, 'Unexpected checkpoint step')
        require(p['gallery_captions_sha256'] == digest(Path(eval_data) / 'gallery_captions.json'),
                'Gallery caption resources differ')
    result['summary'] = calculated
    return result


def candidate(name, result):
    macro = next(r for r in result['summary'] if r['category'] == 'MACRO')
    p = result['provenance']
    return {'name': name, 'optimizer_steps': p['optimizer_steps'],
            'Recall@10': macro['Recall@10'], 'Recall@50': macro['Recall@50'],
            'mean_R': (macro['Recall@10'] + macro['Recall@50']) / 2,
            'checkpoint_sha256': p['checkpoint_sha256'],
            'training_fingerprint': p['training_fingerprint'], 'training_config': p['training_config']}


def choose_original(base, final=None):
    if final is None:
        return base
    require(base['training_fingerprint'] == final['training_fingerprint'], 'Different training runs')
    require(base['training_config'] == final['training_config'], 'Different training configs')
    return final if final['mean_R'] > base['mean_R'] else base


def inspect_checkpoint(path, expected):
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    require('tuning' not in saved and bool(saved.get('history')), 'Expected full original training checkpoint')
    require(len(saved['history']) == saved['history'][-1]['step'] == expected['optimizer_steps'],
            'Checkpoint step differs from evaluation')
    require(saved['fingerprint'] == expected['training_fingerprint'], 'Checkpoint fingerprint differs')
    require(saved['config'] == expected['training_config'], 'Checkpoint config differs')
    require(not any(k.startswith('attention_fusion.') for k in saved['model']), 'Wrong model weights')
    if expected['optimizer_steps'] == 90000:
        require(len(saved['epochs']) == 10 and saved['cursor'] == [10, 0], 'Final checkpoint incomplete')
    info = {'optimizer_steps': len(saved['history']), 'completed_epochs': len(saved['epochs']),
            'cursor': saved['cursor'], 'nonvision_tensors': len(saved['model'])}
    del saved
    return info


def copy_checkpoint(source, destination, expected):
    source, destination = Path(source), Path(destination)
    require(source.resolve() != destination.resolve(), 'Choose a separate snapshot destination')
    before = (source.stat().st_size, source.stat().st_mtime_ns)
    sha = digest(source)
    if expected.get('checkpoint_sha256'):
        require(sha == expected['checkpoint_sha256'], 'Source differs from evaluated checkpoint')
    info = inspect_checkpoint(source, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(digest(destination) == sha, 'Existing destination differs; preserve it and choose a new path')
    else:
        temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
        with source.open('rb') as src, temporary.open('xb') as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        require(digest(temporary) == sha, 'Incomplete checkpoint copy')
        require(before == (source.stat().st_size, source.stat().st_mtime_ns), 'Source changed during copy')
        require(not destination.exists(), 'Destination appeared during copy')
        temporary.rename(destination)
    require(before == (source.stat().st_size, source.stat().st_mtime_ns), 'Source changed during verification')
    inspect_checkpoint(destination, expected)
    receipt = {'source': str(source), 'checkpoint': str(destination), 'checkpoint_sha256': sha,
               **info, 'copy_sha256_verified': True, 'cpu_deserialization_verified': True,
               'note': 'Full model state loading is checked by GPU evaluation; this receipt verifies copying and CPU deserialization.'}
    destination.with_suffix('.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt
