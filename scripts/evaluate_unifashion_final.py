"""Evaluate an original-run snapshot after checking inference code against its fingerprint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fashioniq_evaluation import digest
from src.fashioniq_completion import require


def restore_source_line_endings(path, expected):
    """Restore exact recorded bytes only when the difference is LF versus CRLF."""
    path = Path(path)
    original = path.read_bytes()
    actual = hashlib.sha256(original).hexdigest()
    if actual == expected:
        return False
    lf = original.replace(b'\r\n', b'\n')
    for content in (lf, lf.replace(b'\n', b'\r\n')):
        if hashlib.sha256(content).hexdigest() == expected:
            path.write_bytes(content)
            require(digest(path) == expected, 'Source write verification failed: ' + str(path))
            print('Restored recorded source line endings: ' + str(path), flush=True)
            return True
    raise ValueError(f'Pinned upstream source differs: {path}; expected SHA-256 {expected}, '
                     f'got {actual}. This is not just LF/CRLF. Restore the source from the '
                     'original training environment; do not change download_manifest.json.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('checkpoint', 'eval-data', 'data-dir', 'images-dir', 'output', 'reference-result'):
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--upstream', type=Path, default=ROOT / 'third_party/UniFashion')
    p.add_argument('--assets', type=Path, default=ROOT / 'checkpoints/unifashion')
    a = p.parse_args()
    import torch
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    ref = json.loads(a.reference_result.read_text(encoding='utf-8'))['provenance']
    require(saved['fingerprint'] == ref['training_fingerprint'], 'Wrong original-run fingerprint')
    require(saved['config'] == ref['training_config'], 'Wrong original-run config')
    require(len(saved['history']) == 90000 and len(saved['epochs']) == 10 and saved['cursor'] == [10, 0],
            'Expected completed step90000 checkpoint')
    for name, expected in saved['fingerprint']['code'].items():
        candidates = [ROOT / sub / name for sub in ('src', 'scripts') if (ROOT / sub / name).is_file()]
        require(len(candidates) == 1 and digest(candidates[0]) == expected,
                f'Original training code changed: {name}. Restore the ORIGINAL code ZIP, not current main.')
    require(not any(k.startswith('attention_fusion.') for k in saved['model']), 'This is an attention-fusion checkpoint')
    manifest_path = a.assets / 'download_manifest.json'
    require(digest(manifest_path) == saved['fingerprint']['assets_manifest'],
            'Use the original training assets manifest')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for record in manifest['files']:
        name = record['path'].replace('\\', '/')
        if name.startswith('third_party/UniFashion/src/'):
            path = (a.upstream / name.removeprefix('third_party/UniFashion/')).resolve()
            require(path.is_relative_to(a.upstream.resolve()), 'Unsafe upstream source path')
            restore_source_line_endings(path, record['sha256'])
        if name.startswith('checkpoints/unifashion/bert-base-uncased/'):
            require(digest(a.assets / 'bert-base-uncased' / Path(name).name) == record['sha256'],
                    'Tokenizer/config asset differs: ' + name)
    del saved
    # Import only after establishing the model variant and original source identity.
    from scripts.evaluate_unifashion_fashioniq import evaluate
    evaluate(a)


if __name__ == '__main__':
    main()
