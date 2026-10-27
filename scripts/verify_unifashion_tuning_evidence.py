"""Audit exported tuning evidence without loading weights or rerunning the GPU model."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.fashioniq_evaluation import digest, protocol, summarize
from src.unifashion_tuning import verify_study
from scripts.report_unifashion_tuning import checked_result, choose


def verify(archive, output, full_data, previous):
    archive, output, full_data, previous = map(Path, (archive, output, full_data, previous))
    evidence = output / 'evidence_completed_20261027'
    already_imported = evidence.exists()
    evidence.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Corrupt ZIP')
        for item in z.infolist():
            if not (evidence / item.filename).resolve().is_relative_to(evidence.resolve()):
                raise ValueError('Unsafe archive path')
        if already_imported:
            assert digest(evidence / 'tuning_evidence.zip') == digest(archive)
            for item in z.infolist():
                if not item.is_dir():
                    assert (evidence / item.filename).read_bytes() == z.read(item), item.filename
        else:
            z.extractall(evidence)
    if not already_imported:
        shutil.copy2(archive, evidence / 'tuning_evidence.zip')
    manifest = verify_study(evidence / 'inputs')
    parent = json.loads((evidence / 'parent.json').read_text())
    stages = {'screen': protocol(evidence / 'inputs/screen/evaluation_data'), 'full': protocol(full_data)}
    assert stages['full'][2] == manifest['source_protocol']
    trials = {t['name']: t for t in manifest['config']['trials']}
    checked, tables, frames, checkpoints = [], [], {}, {}
    shared_evaluation_code = None
    training = []
    for name, trial in trials.items():
        folder = evidence / 'trials' / name
        state = json.loads((folder / 'status.json').read_text())
        metrics = pd.read_csv(folder / 'train_metrics.csv')
        assert state['status'] == 'trial_completed'
        assert list(metrics['update']) == list(range(1, len(metrics) + 1))
        assert metrics.samples.eq(trial['batch_size']).all()
        assert np.allclose(metrics.lr, trial['learning_rate'], rtol=0, atol=1e-15)
        assert metrics.samples.sum() == manifest['train_examples'] == 1200
        assert state['tuning']['trial_updates'] == len(metrics)
        assert state['tuning']['parent_sha256'] == parent['checkpoint_sha256']
        assert state['tuning']['study_sha256'] == digest(evidence / 'inputs/study.json')
        assert np.isfinite(metrics.select_dtypes(include='number')).all().all()
        training.append({'trial': name, 'updates': len(metrics), 'examples': int(metrics.samples.sum()),
                         'mean_train_loss': float(np.average(metrics.loss, weights=metrics.samples)),
                         'max_logged_grad_norm_before_clip': float(metrics.grad_norm_before_clip.max())})
    for stage, names in [('screen', ['base', *trials]), ('full', ['base', 'lr1e5_b4'])]:
        queries, galleries, fingerprint = stages[stage]
        lookup = {c: set(ids) for c, ids in galleries.items()}
        keys = ['category', 'candidate', 'target']
        expected = queries.set_index(keys).sort_index()
        for name in names:
            folder = evidence / stage / name
            for method in ('unifashion_image_score', 'unifashion_image_caption_score'):
                result = json.loads((folder / (method + '.json')).read_text())
                p = result['provenance']
                expected_config = dict(parent['config'])
                if name != 'base':
                    expected_config.update(learning_rate=trials[name]['learning_rate'],
                                           batch_size=trials[name]['batch_size'], scheduler='constant', num_workers=0)
                assert p['training_config'] == expected_config and p['precision'] == 'fp32'
                assert p['code_sha256']['unifashion_runtime.py'] == p['training_fingerprint']['code']['unifashion_runtime.py']
                if shared_evaluation_code is None:
                    shared_evaluation_code = p['code_sha256']
                assert p['code_sha256'] == shared_evaluation_code
                checked_result(folder, method, fingerprint, parent['checkpoint_sha256'],
                    trials.get(name), manifest['train_examples'], digest(evidence / 'inputs/study.json'),
                    'full_validation' if stage == 'full' else 'screening_only_reduced_gallery')
                assert p['gallery_captions_sha256'] == digest(
                    (full_data if stage == 'full' else evidence / 'inputs/screen/evaluation_data') / 'gallery_captions.json')
                sha = p['checkpoint_sha256']
                assert checkpoints.setdefault(name, sha) == sha
                frame = pd.read_csv(folder / (method + '_per_query.csv'), keep_default_na=False)
                assert len(frame) == len(queries) and not frame.duplicated(keys).any()
                indexed = frame.set_index(keys).sort_index()
                pd.testing.assert_frame_equal(indexed[expected.columns].astype(str), expected.astype(str))
                for row in frame.to_dict('records'):
                    rank, top = row['target_rank'], json.loads(row['top50'])
                    assert isinstance(rank, int) and 1 <= rank < len(galleries[row['category']])
                    assert len(top) == len(set(top)) == 50 and set(top).issubset(lookup[row['category']])
                    assert row['candidate'] not in top
                    assert (row['target'] in top) == (rank <= 50)
                    if rank <= 50:
                        assert top[rank - 1] == row['target']
                    for k in (10, 50):
                        assert row[f'hit@{k}'] == int(rank <= k)
                actual = summarize(frame.to_dict('records'), galleries)
                for a, b in zip(actual, result['summary'], strict=True):
                    for key in a:
                        if key.startswith('Recall'):
                            assert np.isclose(a[key], b[key], rtol=0, atol=1e-10)
                        else:
                            assert a[key] == b[key]
                tables += [{'stage': stage, 'checkpoint': name, 'method': method, **r} for r in actual]
                frames[(stage, name, method)] = indexed
                checked.append({'stage': stage, 'checkpoint': name, 'method': method, 'queries_verified': len(frame)})
    table = pd.DataFrame(tables)
    table.to_csv(output / 'verified_recall.csv', index=False)
    pd.DataFrame(training).to_csv(output / 'verified_training_summary.csv', index=False)
    primary = table[(table.method == 'unifashion_image_caption_score') & (table.category == 'MACRO')].copy()
    primary['mean_R'] = (primary['Recall@10'] + primary['Recall@50']) / 2
    observed = choose([{'name': r['checkpoint'], 'mean_R': r['mean_R']}
                       for r in primary[primary.stage == 'screen'].to_dict('records')])
    assert observed == json.loads((evidence / 'report/selection.json').read_text())['selected']
    for stage, filename in [('screen', 'screening.csv'), ('full', 'full_validation.csv')]:
        exported = pd.read_csv(evidence / 'report' / filename).set_index('name')
        calculated = primary[primary.stage == stage].set_index('checkpoint')
        assert set(exported.index) == set(calculated.index)
        for name in exported.index:
            for metric, local in [('R@10', 'Recall@10'), ('R@50', 'Recall@50'), ('mean_R', 'mean_R')]:
                assert np.isclose(exported.loc[name, metric], calculated.loc[name, local], atol=1e-10, rtol=0)
    older = json.loads(previous.read_text())
    assert older['protocol'] == stages['full'][2]
    base_result = json.loads((evidence / 'full/base/unifashion_image_caption_score.json').read_text())
    assert older['provenance']['training_fingerprint'] == base_result['provenance']['training_fingerprint']
    candidates = []
    old_macro = next(r for r in older['summary'] if r['category'] == 'MACRO')
    for label, r, sha in [('step29500', old_macro, older['provenance']['checkpoint_sha256'])] + [
            (r['checkpoint'], r, checkpoints[r['checkpoint']]) for r in primary[primary.stage == 'full'].to_dict('records')]:
        candidates.append({'name': label, 'Recall@10': r['Recall@10'], 'Recall@50': r['Recall@50'],
                           'mean_R': (r['Recall@10'] + r['Recall@50']) / 2, 'checkpoint_sha256': sha})
    pd.DataFrame(candidates).to_csv(output / 'checkpoint_comparison.csv', index=False)
    best = max(candidates, key=lambda r: r['mean_R'])
    decision = {'selection_date': '2026-10-27', 'criterion': 'mean of full-validation macro Recall@10 and Recall@50',
        'scoring_method': 'unifashion_image_caption_score', 'selected': best,
        'scope': 'best among the three listed full-validation checkpoints; not all training epochs',
        'screening_shortlist': observed, 'full_validation_decision': 'retain_base',
        'original_snapshot_on_drive': 'MyDrive/Thesis/UniFashion/week8/tuning/fp32_lr_batch_01/parent_snapshot.pt',
        'checkpoint_weights_received': False, 'dedicated_best_checkpoint_copy_verified': False,
        'full_training_complete': False, 'latest_training_progress': 'User reports epoch 9 step 78000; not evidenced by this ZIP'}
    (output / 'best_checkpoint_selection.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
    transitions = []
    for method in ('unifashion_image_score', 'unifashion_image_caption_score'):
        a, b = frames[('full', 'base', method)], frames[('full', 'lr1e5_b4', method)]
        for k in (10, 50):
            x, y = a[f'hit@{k}'].astype(bool), b[f'hit@{k}'].astype(bool)
            transitions.append({'method': method, 'k': k, 'both_hit': int((x & y).sum()),
                'trial_only_hit': int((~x & y).sum()), 'base_only_hit': int((x & ~y).sum()), 'both_miss': int((~x & ~y).sum())})
    pd.DataFrame(transitions).to_csv(output / 'full_validation_transitions.csv', index=False)
    audit = {'verified_on': '2026-10-27', 'archive_sha256': digest(archive), 'archive_bytes': archive.stat().st_size,
        'evaluations_verified': checked, 'trial_metrics_verified': training,
        'full_data_protocol': stages['full'][2],
        'limits': 'Verifies exported ranks, summaries and provenance consistency; no model weights or score matrices received, no independent GPU rerun. Current local attention-fusion runtime differs from the original runtime recorded by these runs.',
        'checks': ['ZIP CRC and study input hashes', 'Complete unique queries and exact text', 'Valid target ranks and saved top50',
                   'Reference exclusion in top50', 'Per-category/macro/micro recomputed', 'Same parent/study/budget across trials',
                   'Same checkpoint hashes between screening and full evaluation', 'Original training fingerprint matches step29500 evidence',
                   'Reported selection and comparison tables recomputed',
                   'Shared evaluation code hashes and expected FP32 training configs'],
        'files_sha256': {p.relative_to(evidence).as_posix(): digest(p) for p in sorted(evidence.rglob('*')) if p.is_file()}}
    (output / 'full_evidence_verification.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({'best': best, 'transitions': transitions, 'checked_results': len(checked)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full-data', type=Path, default=ROOT / 'logs/week41_fashioniq_evaluation/evaluation_data')
    parser.add_argument('--previous', type=Path, default=ROOT / 'logs/week41_fashioniq_evaluation/evidence_fp32_step29500/results/unifashion_image_caption_score.json')
    args = parser.parse_args()
    verify(args.archive, args.output, args.full_data, args.previous)
