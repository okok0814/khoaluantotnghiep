"""Create an offline visual review sheet from reproducibly sampled retrieval outcomes."""
import argparse
import base64
import html
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--data-zip',type=Path,default=ROOT/'checkpoints/unifashion_week8_data.zip')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    samples=pd.read_csv(args.samples,keep_default_na=False).to_dict('records')
    images={}
    with zipfile.ZipFile(args.data_zip) as z:
        manifest=json.loads(z.read('data/images_manifest.json'))
        for row in samples:
            ids=[row['candidate'],row['target']]+json.loads(row['top50_A'])[:3]+json.loads(row['top50_B'])[:3]
            for name in ids:
                if name not in images:
                    content=z.read('images/'+manifest[name]['file'])
                    images[name]='data:image/jpeg;base64,'+base64.b64encode(content).decode('ascii')
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><title>FashionIQ error review</title>',
          '<style>body{font:16px system-ui;max-width:1200px;margin:24px auto}article{border-top:1px solid #ccc;padding:20px 0}.grid{display:flex;flex-wrap:wrap;gap:12px}figure{width:125px;margin:0}img{width:125px;height:150px;object-fit:contain}figcaption{font-size:12px}textarea{width:95%;height:65px}select,button{padding:8px;margin:8px 0}</style>',
          '<h1>FashionIQ retrieval review</h1><p>Seed-42 samples from each category and outcome. A miss means the labeled target is outside top 10; other retrieved products may also satisfy the request. Images alone may not establish material.</p>',
          '<p>Record visible evidence. Suggested labels: color, shape/length, pattern, category/style, multiple attributes, plausible alternative target, ambiguous description, or uncertain. Leave unsupported judgments blank.</p>',
          '<button onclick="downloadLabels()">Download annotations</button>']
    for i,row in enumerate(samples):
        h=html.escape
        body.append(f'<article data-row="{i}"><h2>{i+1}. {h(row["category"])} / {h(row["outcome"])}</h2><p>{h(row["text"])}</p>')
        body.append(f'<p>A: {h(row["system_A"])} (target rank {row["target_rank_A"]}); B: {h(row["system_B"])} (rank {row["target_rank_B"]})</p><div class="grid">')
        figures=[('Reference',row['candidate']),('Labeled target',row['target'])]
        figures += [(f'A #{k+1}',n) for k,n in enumerate(json.loads(row['top50_A'])[:3])]
        figures += [(f'B #{k+1}',n) for k,n in enumerate(json.loads(row['top50_B'])[:3])]
        for label,name in figures:
            body.append(f'<figure><img data-image="{h(name)}" alt="{h(label)}"><figcaption>{h(label)}<br>{h(name)}</figcaption></figure>')
        body.append(f'</div><label>Observed issue <input id="label{i}"></label><br><textarea id="note{i}" placeholder="Evidence or uncertainty"></textarea></article>')
    # Escape script closers in dataset text before embedding JSON.
    encoded=json.dumps({'samples':samples,'images':images}).replace('<','\\u003c')
    body.append('<script>const data='+encoded+'; document.querySelectorAll("img[data-image]").forEach(el=>el.src=data.images[el.dataset.image]);\n'
                'const key="fashioniq-review-"+data.samples.length; let saved=JSON.parse(localStorage.getItem(key)||"{}");\n'
                'document.querySelectorAll("input,textarea").forEach(el=>{el.value=saved[el.id]||"";el.oninput=()=>{saved[el.id]=el.value;localStorage.setItem(key,JSON.stringify(saved));};});\n'
                'function downloadLabels(){let rows=data.samples.map((r,i)=>({...r,human_error_type:document.getElementById("label"+i).value,reviewer_note:document.getElementById("note"+i).value}));let a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{type:"application/json"}));a.download="error_annotations.json";a.click();}\n</script></html>')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(body),encoding='utf-8')
    print(f'Visual review: {len(samples)} samples, {len(images)} distinct images, {args.output}')


if __name__=='__main__':
    main()
