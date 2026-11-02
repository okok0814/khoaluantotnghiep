"""Build an additive Colab evaluation bundle using the original training code ZIP."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fashioniq_evaluation import digest
from scripts.download_unifashion import SOURCE_FILES


def main():
    evidence = ROOT / 'logs/week43_44_unifashion_completion'
    evaluation = ROOT / 'logs/week41_fashioniq_evaluation'
    base = ROOT / 'logs/week42_unifashion_tuning/evidence_completed_20261027/full/base'
    old_code = ROOT / 'checkpoints/unifashion_week8_code.zip'
    reference = json.loads((base / 'unifashion_image_caption_score.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(old_code) as z:
        for name, sha in reference['provenance']['training_fingerprint']['code'].items():
            candidates = [p for p in z.namelist() if Path(p).name == name]
            if len(candidates) != 1 or hashlib.sha256(z.read(candidates[0])).hexdigest() != sha:
                raise ValueError('Original code ZIP differs from evaluated run: ' + name)
    inputs = ROOT / '.cache/unifashion_completion_inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    for subdir in ('evaluation_data', 'baselines'):
        (inputs / subdir).mkdir(exist_ok=True)
        for p in (evaluation / subdir).iterdir():
            if p.is_file():
                shutil.copyfile(p, inputs / subdir / p.name)
    shutil.copyfile(evaluation / 'preparation.json', inputs / 'preparation.json')
    (inputs / 'base').mkdir(exist_ok=True)
    for name in ('status.json', 'unifashion_image_score.json', 'unifashion_image_score_per_query.csv',
                 'unifashion_image_caption_score.json', 'unifashion_image_caption_score_per_query.csv'):
        shutil.copyfile(base / name, inputs / 'base' / name)
    for name in ('training_verification.json', 'training_epochs.csv'):
        shutil.copyfile(evidence / name, inputs / name)
    shutil.copyfile(old_code, inputs / 'original_code.zip')
    files = ['src/fashioniq_evaluation.py', 'src/fashioniq_completion.py', 'src/unifashion_tuning.py',
             'src/attention_fusion.py', 'scripts/evaluate_unifashion_fashioniq.py',
             'scripts/evaluate_unifashion_final.py', 'scripts/finalize_unifashion_checkpoint.py',
             'scripts/report_unifashion_completion.py', 'tests/test_fashioniq_completion.py',
             'tests/test_fashioniq_evaluation.py']
    content = {name: ROOT / name for name in files}
    for name in SOURCE_FILES:
        if not name.startswith('dataset/'):
            p = ROOT / 'third_party/UniFashion' / name
            if not p.is_file():
                raise FileNotFoundError('Missing pinned upstream source: ' + str(p))
            content['third_party/UniFashion/' + name] = p
    content.update({'completion_inputs/' + p.relative_to(inputs).as_posix(): p
                    for p in inputs.rglob('*') if p.is_file()})
    manifest = {'files': {name: digest(p) for name, p in sorted(content.items())},
                'preserves_original_training_source': True, 'contains_weights': False, 'contains_images': False}
    output = ROOT / 'checkpoints/unifashion_completion.zip'
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, p in content.items():
            z.write(p, name)
        z.writestr('completion_manifest.json', json.dumps(manifest, indent=2))
    receipt = {'bundle': str(output), 'bytes': output.stat().st_size, 'sha256': digest(output), **manifest}
    (evidence / 'bundle.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ('bundle', 'bytes', 'sha256')}, indent=2))


if __name__ == '__main__':
    main()
