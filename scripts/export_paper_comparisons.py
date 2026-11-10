"""Export lightweight comparison evidence; exclude weights, features and resumable caches."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.fashioniq_evaluation import digest


def export(results, output):
    files = [p for p in results.rglob("*") if p.is_file()
             and not set(p.relative_to(results).parts) & {"cache", "chunks", "features", "assets"}
             and p.suffix in (".json", ".csv", ".md", ".tex", ".xml", ".png", ".svg")]
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,p.relative_to(results).as_posix())
        z.writestr("export_manifest.json",json.dumps({"files": {p.relative_to(results).as_posix():digest(p) for p in files},
             "contains_weights":False, "note":"Reports mark incomplete experiments as pending; export does not imply completion."},indent=2))
    print(f"Exported {len(files)} evidence files: {output}")


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    export(args.results,args.output)
