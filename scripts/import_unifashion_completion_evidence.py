"""Verify a Colab evidence ZIP, preserve its contents, and regenerate the report."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fashioniq_completion import candidate, choose_original, read_result, require
from src.fashioniq_evaluation import digest
from scripts.report_unifashion_completion import generate


def plot_recall(results, output):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['FashionCLIP\nadditive', 'UniFashion 60k\n+ gallery caption',
              'UniFashion 90k\n+ gallery caption', 'UniFashion 90k\nno caption score']
    metrics = [next(s for s in r['summary'] if s['category'] == 'MACRO') for r in results]
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    for offset, key, color in [(-0.19, 'Recall@10', '#2166ac'), (0.19, 'Recall@50', '#d95f02')]:
        bars = ax.bar([i + offset for i in range(4)], [m[key] for m in metrics],
                      width=0.36, label=key, color=color)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    ax.set(xticks=range(4), xticklabels=labels, ylim=(0, 100), ylabel='Macro Recall (%)',
           title='FashionIQ validation: 6,016 queries, three categories')
    ax.legend(loc='upper left', ncols=2)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.2)
    fig.text(0.5, -0.025, 'Attention-fusion pending. These are validation results, not official test scores.',
             ha='center', fontsize=9)
    for extension in ('png', 'svg'):
        fig.savefig(output / ('recall_comparison.' + extension), dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--received-on', required=True, help='Receipt date, YYYY-MM-DD')
    parser.add_argument('--inputs', type=Path, default=ROOT / '.cache/unifashion_completion_inputs')
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/week43_44_unifashion_completion')
    args = parser.parse_args()
    from datetime import date
    received = date.fromisoformat(args.received_on)
    destination = args.output / ('evidence_completed_' + received.strftime('%Y%m%d'))
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        require(archive.testzip() is None, 'ZIP CRC failed')
        require(len(archive.namelist()) == len(set(archive.namelist())), 'Duplicate ZIP entries')
        for item in archive.infolist():
            name = PurePosixPath(item.filename)
            require(not item.is_dir() and not name.is_absolute() and '..' not in name.parts
                    and ':' not in item.filename and '\\' not in item.filename,
                    'Unexpected archive path: ' + item.filename)
            require(name.suffix in {'.json', '.csv', '.xml', '.md', '.png', '.log'}, 'Unexpected file type')
            path = destination / item.filename
            require(path.resolve().is_relative_to(destination.resolve()), 'Archive path escaped destination')
            content = archive.read(item)
            if path.exists():
                require(path.read_bytes() == content, 'Existing evidence differs: ' + str(path))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

    def read(name):
        return json.loads((destination / name).read_text(encoding='utf-8'))

    data = args.inputs / 'evaluation_data'
    methods = ('unifashion_image_caption_score', 'unifashion_image_score')
    final = [read_result(destination / 'final_results', method, data, 'original', 90000) for method in methods]
    baseline = read_result(args.inputs / 'baselines', 'fashionclip_additive', data)
    base = read_result(args.inputs / 'base', methods[0], data, 'original', 60000)
    selected = choose_original(candidate('step60000', base), candidate('step90000', final[0]))
    selection = read('report/checkpoint_selection.json')
    require(selection['selected'] == selected and selection['final_checkpoint_evaluated'] is True,
            'Exported checkpoint selection differs from recalculated scores')
    provenance = read('final_results/provenance.json')
    require(final[0]['provenance'] == final[1]['provenance'], 'Scoring provenance differs')
    require({k: v for k, v in final[0]['provenance'].items() if k != 'evaluation_seconds'} == provenance,
            'Saved provenance differs from results')
    require(provenance['completed_epochs'] == 10 and provenance['planned_epochs'] == 10
            and provenance['completed_batches_in_current_epoch'] == 0, 'Final evaluation progress differs')
    for method in methods:
        partial = destination / 'final_results' / (method + '_partial.csv')
        if partial.exists():
            require(digest(partial) == digest(partial.with_name(method + '_per_query.csv')),
                    'Partial and final CSV differ')
    # Cross-check against the previously verified complete training log.
    for exported, original in [('training_status.json', 'status.json'), ('training_config.json', 'config.json'),
                               ('training_environment.json', 'environment.json')]:
        reference = json.loads((args.output / 'training_evidence' / original).read_text(encoding='utf-8'))
        require(read(exported) == reference, 'Training metadata differs: ' + exported)
    pd.testing.assert_frame_equal(pd.read_csv(destination / 'training_epoch_metrics.csv'),
                                  pd.read_csv(args.output / 'training_evidence/epoch_metrics.csv'))
    require(digest(destination / 'training_training_curves.png') ==
            digest(args.output / 'training_evidence/training_curves.png'), 'Training plot differs')
    snapshot, receipt = read('fp32_step90000.json'), read('report/checkpoint_copy.json')
    for r, sha, step in [(snapshot, provenance['checkpoint_sha256'], 90000),
                         (receipt, selected['checkpoint_sha256'], selected['optimizer_steps'])]:
        require(r['checkpoint_sha256'] == sha and r['optimizer_steps'] == step,
                'Copy receipt identity differs')
        require(r['copy_sha256_verified'] is True and r['cpu_deserialization_verified'] is True,
                'Copy/reload receipt incomplete')
        if step == 90000:
            require(r['completed_epochs'] == 10 and r['cursor'] == [10, 0], 'Copy progress differs')
    require(receipt['selection'] == selection, 'Copy receipt selection differs')
    require(Path(receipt['checkpoint']).name == f"best_recall_{selected['name']}.pt", 'Unexpected best filename')
    if selected['optimizer_steps'] == 90000:
        require(receipt['source'] == snapshot['checkpoint'], 'Snapshot-to-best copy chain differs')
    suites = ET.parse(destination / 'completion_tests.xml').getroot().findall('testsuite')
    tests = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
             for key in ('tests', 'failures', 'errors', 'skipped')}
    require(tests['tests'] > 0 and not any(tests[k] for k in ('failures', 'errors', 'skipped')),
            'Exported completion tests did not all pass')
    # Raw files above remain immutable; the top-level report is a regenerated view.
    record = {'received_on': args.received_on, 'archive_name': args.archive.name,
              'archive_sha256': digest(args.archive), 'archive_bytes': args.archive.stat().st_size,
              'evidence_directory': destination.name,
              'files': {p.relative_to(destination).as_posix(): digest(p) for p in sorted(destination.rglob('*')) if p.is_file()},
              'full_validation_queries': next(r['queries'] for r in final[0]['summary'] if r['category'] == 'MACRO'),
              'both_scoring_variants_recomputed': True, 'baseline_ranks_rechecked': bool(baseline),
              'original_training_metadata_matched': True, 'receipt_chain_verified': True,
              'exported_tests': tests, 'weights_received': False, 'independent_gpu_inference_run': False}
    (args.output / 'evidence_import.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    shutil.copyfile(destination / 'report/checkpoint_copy.json', args.output / 'checkpoint_copy.json')
    plot_recall([baseline, base, final[0], final[1]], args.output)
    state = generate(args.inputs, args.output, destination / 'final_results')
    require(json.loads((args.output / 'checkpoint_selection.json').read_text(encoding='utf-8')) == selection,
            'Regenerated selection differs')
    require(state == read('report/status.json'), 'Regenerated completion state differs')
    for name in ('checkpoint_comparison.csv', 'model_comparison.csv'):
        pd.testing.assert_frame_equal(pd.read_csv(args.output / name), pd.read_csv(destination / 'report' / name))
    print(json.dumps({'verification': 'passed', 'selected': selected['name'], 'state': state}, indent=2))


if __name__ == '__main__':
    main()
