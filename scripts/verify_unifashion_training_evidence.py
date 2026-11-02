"""Verify exported completed-training logs; never infer unmeasured retrieval scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.fashioniq_evaluation import digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(archive, output, reference, data_bundle):
    archive, output, reference = map(Path, (archive, output, reference))
    evidence = output / 'training_evidence'
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None, 'ZIP CRC failure')
        names = z.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP members')
        for name in names:
            require('/' not in name and '\\' not in name and name not in ('.', '..'),
                    'Expected flat training evidence ZIP')
            require(Path(name).suffix in {'.json', '.jsonl', '.csv', '.log', '.png', '.md', '.xml'},
                    'Unexpected evidence file type')
        # Check every existing member before writing anything; preserve evidence.
        for name in names:
            dest = evidence / name
            require(not dest.exists() or dest.read_bytes() == z.read(name),
                    f'Existing evidence differs: {name}')
        evidence.mkdir(parents=True, exist_ok=True)
        for name in names:
            if not (evidence / name).exists():
                (evidence / name).write_bytes(z.read(name))
    read = lambda name: json.loads((evidence / name).read_text(encoding='utf-8'))
    status, cfg, env, audit = [read(n) for n in
                             ('status.json', 'config.json', 'environment.json', 'data_audit.json')]
    prior = json.loads(reference.read_text(encoding='utf-8'))['provenance']
    require(env['fingerprint'] == prior['training_fingerprint'], 'Not the original evaluated training run')
    require(cfg == prior['training_config'], 'Training configuration differs from the evaluated run')
    # The runner reserializes JSON (Windows CRLF -> Linux LF). Verify source bytes
    # against the fingerprint and compare the exported copy semantically.
    with zipfile.ZipFile(data_bundle) as z:
        original_audit = z.read('data/data_audit.json')
    require(hashlib.sha256(original_audit).hexdigest() == env['fingerprint']['data_audit'],
            'Original data audit hash differs')
    require(json.loads(original_audit) == audit, 'Exported data audit contents differ')
    require(status['status'] == 'completed', 'Training not completed')
    require(status['completed_epochs'] == status['planned_epochs'] == cfg['epochs'] == 10,
            'Expected 10 completed epochs')
    steps = pd.read_csv(evidence / 'train_metrics.csv')
    epochs = pd.read_csv(evidence / 'epoch_metrics.csv')
    plans = read('batch_plan.json')
    require(len(steps) == status['optimizer_steps'] == status['checkpoint_step'] == 90000,
            'Expected 90000 saved optimizer updates')
    require(np.array_equal(steps.step, np.arange(1, 90001)), 'Noncontiguous/duplicate steps')
    require(np.isfinite(steps.select_dtypes(include='number').to_numpy()).all(), 'Nonfinite training log')
    require(np.isfinite(epochs.select_dtypes(include='number').to_numpy()).all(), 'Nonfinite epoch log')
    require(np.array_equal(epochs.epoch, np.arange(1, 11)), 'Incomplete epoch history')
    require(len(plans) == 10, 'Incomplete batch plan')
    losses = ('loss_itc', 'loss_itm', 'loss_ttc')
    require(np.allclose(steps.loss, steps[list(losses)].sum(axis=1), atol=1e-6, rtol=1e-5),
            'Batch loss components do not sum')
    for plan, row in zip(plans, epochs.to_dict('records'), strict=True):
        epoch = plan['epoch']
        part = steps[steps.epoch == epoch]
        require(row['epoch'] == epoch and len(part) == plan['batches'] == 9000, 'Epoch batch count differs')
        require(np.array_equal(part.batch, np.arange(1, 9001)), 'Epoch batch order differs')
        require((part.samples == cfg['batch_size']).all(), 'Unexpected batch size')
        require(part.samples.sum() == row['train_samples'] == plan['samples'] == 18000,
                'Epoch example coverage differs')
        require(row['step'] == int(part.step.iloc[-1]) == epoch * 9000, 'Epoch boundary differs')
        require(row['dropped_samples'] == len(plan['dropped_row_indices']) == 0, 'Dropped queries')
        require(row['monitor_samples'] == audit['monitor']['rows'] == 96, 'Monitor coverage differs')
        for loss in ('loss', *losses):
            mean = np.average(part[loss], weights=part.samples)
            require(np.isclose(mean, row['train_' + loss], atol=1e-10), 'Epoch mean mismatch: ' + loss)
        require(np.isclose(row['monitor_loss'], sum(row['monitor_' + k] for k in losses)),
                'Monitor components do not sum')
    best = epochs.loc[epochs.monitor_loss.idxmin()]
    require(np.isclose(best.monitor_loss, status['best_monitor_loss']), 'Best monitor selection differs')
    require((steps.lr > 0).all() and (steps.lr <= cfg['learning_rate'] * (1 + 1e-6)).all(),
            'LR outside configured range')
    tests = ET.parse(evidence / 'tests_gpu_environment.xml').getroot()
    suites = list(tests.iter('testsuite'))
    test_counts = {k: sum(int(s.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    result = {
        'archive_sha256': digest(archive), 'archive_bytes': archive.stat().st_size,
        'training_completed': True, 'epochs': 10, 'optimizer_steps': 90000,
        'examples_processed': int(steps.samples.sum()),
        'best_monitor_epoch': int(best.epoch), 'best_monitor_loss': float(best.monitor_loss),
        'final_train_loss': float(epochs.iloc[-1].train_loss),
        'final_monitor_loss': float(epochs.iloc[-1].monitor_loss),
        'recorded_update_hours': float(steps.seconds.sum() / 3600),
        'update_time_excludes': 'data loading outside train_batch timer, setup, validation, checkpoint I/O and replayed work',
        'amp_retries': int(steps.amp_retries.sum()),
        'max_grad_norm_before_clip': float(steps.grad_norm_before_clip.max()),
        'reported_peak_allocated_gib': status['peak_cuda_allocated_bytes'] / 2**30,
        'gpu': env['gpu'], 'tests_from_exported_xml': test_counts,
        'fingerprint': env['fingerprint'], 'training_config': cfg,
        'final_checkpoint_recall_evaluated': False, 'checkpoint_weights_received': False,
        'checkpoint_copy_reload_verified': False,
        'limits': 'Checks exported log consistency, not GPU execution or checkpoint file contents. '
                  'Historical Recall at steps 29500/60000 remains separate from final-checkpoint evaluation.',
        'checks': ['ZIP CRC', 'original-run fingerprint/config', '90000 consecutive finite updates',
                   '18000 examples and 9000 batches per epoch', 'recomputed epoch means',
                   'monitor selection', 'recorded learning-rate bounds'],
        'files_sha256': {p.name: digest(p) for p in sorted(evidence.iterdir()) if p.is_file()},
    }
    (output / 'training_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    epochs.to_csv(output / 'training_epochs.csv', index=False)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=ROOT / 'logs/week42_unifashion_tuning/evidence_completed_20261027/full/base/unifashion_image_caption_score.json')
    p.add_argument('--data-bundle', type=Path, default=ROOT / 'checkpoints/unifashion_week8_data.zip')
    a = p.parse_args()
    r = verify(a.archive, a.output, a.reference, a.data_bundle)
    print(json.dumps({k: r[k] for k in ('training_completed', 'epochs', 'optimizer_steps', 'best_monitor_epoch',
                                      'best_monitor_loss', 'final_train_loss', 'final_monitor_loss')}, indent=2))
