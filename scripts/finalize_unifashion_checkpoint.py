"""Create immutable final snapshots and preserve the checkpoint selected by validation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fashioniq_completion import candidate, copy_checkpoint, require


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('snapshot')
    s.add_argument('--source', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--reference-result', type=Path, required=True)
    s = sub.add_parser('save-best')
    s.add_argument('--selection', type=Path, required=True)
    s.add_argument('--base', type=Path, required=True)
    s.add_argument('--final', type=Path, required=True)
    s.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'snapshot':
        reference = json.loads(a.reference_result.read_text(encoding='utf-8'))
        expected = candidate('step90000', reference)
        expected.update(optimizer_steps=90000, checkpoint_sha256=None)
        receipt = copy_checkpoint(a.source, a.output, expected)
    else:
        selection = json.loads(a.selection.read_text(encoding='utf-8'))
        require(selection['final_checkpoint_evaluated'], 'Evaluate the final checkpoint before final selection')
        selected = selection['selected']
        require(selected['name'] in ('step60000', 'step90000'), 'Unknown selected candidate')
        source = a.base if selected['name'] == 'step60000' else a.final
        destination = a.output_dir / ('best_recall_' + selected['name'] + '.pt')
        receipt = copy_checkpoint(source, destination, selected)
        receipt['selection'] = selection
        (a.selection.parent / 'checkpoint_copy.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
