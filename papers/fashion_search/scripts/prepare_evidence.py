"""Verify the supplied prediction archive and derive manuscript tables without inference."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.fashioniq_completion import read_result
from src.fashioniq_evaluation import digest, protocol
from src.paper_comparison import paired_bootstrap

PAPER = ROOT / 'papers/fashion_search'
EVAL = ROOT / 'logs/week41_fashioniq_evaluation/evaluation_data'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    sha = digest(args.archive)
    unpack = ROOT / '.cache/manuscript_review' / ('evidence_' + sha[:16])
    unpack.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), 'Duplicate archive entries'
        for name in names:
            path = PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts and '\\' not in name and ':' not in name
        manifest = json.loads(archive.read('results/export_manifest.json'))['files']
        for name, expected in manifest.items():
            assert hashlib.sha256(archive.read('results/' + name)).hexdigest() == expected, name
        # Validate paths before extracting; preserve the ZIP and historical evidence unchanged.
        for name in names:
            if name.endswith('/'):
                continue
            dest = unpack / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            content = archive.read(name)
            if dest.exists():
                assert dest.read_bytes() == content, str(dest)
            else:
                dest.write_bytes(content)
    out = PAPER / 'evidence'
    out.mkdir(parents=True, exist_ok=True)
    sources, frames, records, verified, excluded = {}, {}, [], [], []
    for path in sorted((unpack / 'results').rglob('*_per_query.csv')):
        method = path.name.removesuffix('_per_query.csv')
        result = read_result(path.parent, method, EVAL)
        verified.append(method)
        if 'gate' in method:
            excluded.append(method)
            continue
        sources[method] = {'path': path.relative_to(unpack).as_posix(),
                           'sha256': digest(path), 'result_sha256': digest(path.with_name(method + '.json')),
                           'provenance': result['provenance']}
        frames[method] = pd.read_csv(path, keep_default_na=False)
        records.extend({'method': method, **row} for row in result['summary'])
    historical = [
        ('final', 'logs/week43_44_unifashion_completion/evidence_completed_20261102/final_results', 90000),
        ('step60000', 'logs/week42_unifashion_tuning/evidence_completed_20261027/full/base', 60000),
        ('step29500', 'logs/week41_fashioniq_evaluation/evidence_fp32_step29500/results', 29500),
        ('tuned', 'logs/week42_unifashion_tuning/evidence_completed_20261027/full/lr1e5_b4', None),
    ]
    for tag, folder, step in historical:
        for score in ('image_score', 'image_caption_score'):
            method = 'unifashion_' + score
            path = ROOT / folder
            result = read_result(path, method, EVAL, variant=None if tag == 'step29500' else 'original', expected_steps=step)
            if tag == 'step29500':
                # This older schema predates evaluation_scope. read_result above still
                # enforces all 6,016 queries and the complete official gallery protocol.
                p = result['provenance']
                status = json.loads((path / 'status.json').read_text())
                assert p['optimizer_steps'] == 29500
                assert status['status'] == 'evaluation_completed'
                assert status['checkpoint_sha256'] == p['checkpoint_sha256']
                assert p['code_sha256']['unifashion_runtime.py'] == 'ab27b7634b40bf271957f67311a775bc7d976184311676dcd957eda875ceed66'
                assert p['training_fingerprint']['code']['unifashion_runtime.py'] == p['code_sha256']['unifashion_runtime.py']
            label = tag + '_' + method
            csv = path / (method + '_per_query.csv')
            frames[label] = pd.read_csv(csv, keep_default_na=False)
            records.extend({'method': label, **row} for row in result['summary'])
            sources[label] = {'path': csv.relative_to(ROOT).as_posix(), 'sha256': digest(csv),
                              'result_sha256': digest(path / (method+'.json')), 'provenance': result['provenance']}
    table = pd.DataFrame(records)
    table.to_csv(out / 'verified_results.csv', index=False)
    seed_rows = []
    for family in ('weighted', 'mlp'):
        group = table[table.method.str.startswith('fashionclip_'+family+'_seed')]
        for category in ('dress','shirt','toptee','MACRO','MICRO'):
            part = group[group.category == category]
            seed_rows.append({'family':family,'category':category,'seeds':len(part), **{
                f'{metric}_{stat}': float(part[metric].mean() if stat=='mean' else part[metric].std(ddof=1))
                for metric in ('Recall@10','Recall@50') for stat in ('mean','sd')}})
    pd.DataFrame(seed_rows).to_csv(out / 'seed_summary.csv', index=False)
    pairs = [
        ('fashionclip_text_only_matched','fashionclip_additive_matched'),
        ('fashionclip_mlp_seed42','fashionclip_text_only_matched'),
        ('sprc_vitl','final_unifashion_image_score'),
        ('final_unifashion_image_score','clip4cir_combiner'),
        ('final_unifashion_image_caption_score','sprc_vitl'),
        ('final_unifashion_image_caption_score','final_unifashion_image_score'),
        ('final_unifashion_image_score','unifashion_initial_image_score'),
        ('final_unifashion_image_caption_score','unifashion_initial_image_caption_score'),
        ('final_unifashion_image_caption_score','step60000_unifashion_image_caption_score'),
        ('tuned_unifashion_image_caption_score','step60000_unifashion_image_caption_score'),
    ]
    intervals, transitions = [], []
    for a,b in pairs:
        intervals.extend({'A':a,'B':b,**row} for row in paired_bootstrap(frames[a], frames[b], repeats=5000))
        aa = frames[a].set_index(['category','candidate','target']).sort_index()
        bb = frames[b].set_index(['category','candidate','target']).sort_index()
        for k in (10,50):
            ah, bh = aa.target_rank <= k, bb.target_rank <= k
            transitions.append({'A':a,'B':b,'k':k,'both_hit':int((ah&bh).sum()),
                                'A_only':int((ah&~bh).sum()),'B_only':int((~ah&bh).sum()),
                                'both_miss':int((~ah&~bh).sum())})
    pd.DataFrame(intervals).to_csv(out / 'paired_intervals.csv', index=False)
    pd.DataFrame(transitions).to_csv(out / 'query_transitions.csv', index=False)
    # Archive-derived summaries are independently reconstructed rather than trusted.
    exported = pd.read_csv(unpack / 'report/measured_results.csv')
    for method in verified:
        if method in excluded:
            continue
        a = table[table.method == method].set_index('category')
        b = exported[exported.method == method].set_index('category')
        assert np.allclose(a[['Recall@10','Recall@50']], b.loc[a.index,['Recall@10','Recall@50']], atol=1e-10, rtol=0)
    q, galleries, fingerprint = protocol(EVAL)
    verification = {'archive':args.archive.name,'archive_sha256':sha,'manifest_files_checked':len(manifest),
                    'verified_new_methods':verified,'excluded_teammate_methods':excluded,
                    'included_method_snapshots':len(frames),'queries_per_method':len(q),
                    'galleries':{k:len(v) for k,v in galleries.items()},'protocol':fingerprint,
                    'bootstrap_replicates':5000,'bootstrap_seed':42,
                    'verification_scope':'Hashes, complete query/gallery coverage, ranks/top50/hits, summaries; no new GPU inference',
                    'sources':sources}
    (out / 'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in verification.items() if k not in ('sources','protocol')},indent=2))
    print(table[table.category=='MACRO'][['method','Recall@10','Recall@50']].to_string(index=False))


if __name__ == '__main__':
    main()
