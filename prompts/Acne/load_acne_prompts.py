"""Load DermaLens acne prompts without reconstructing or rewriting their text.

Examples:
  python load_acne_prompts.py --source ./DermaLens_SD21_Prompts --split train --limit 2
  python load_acne_prompts.py --source ./DermaLens_Acne_300_Prompts_JSON_TXT_GitHub_Ready.zip --split all --limit 3

API:
  prompts = load_prompts(GITHUB_RAW_MASTER_JSON_URL)
  for job in iter_generation_jobs(prompts, split='train'): ...
"""
from __future__ import annotations
import csv, io, json, re, zipfile, argparse
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

MASTER_PATH = 'DermaLens_SD21_Prompts/01_Acne/json/acne_all_300.json'

def _read_url(url: str) -> bytes:
    with urlopen(Request(url, headers={'User-Agent': 'DermaLens-PromptLoader/1.0'}), timeout=90) as resp:
        return resp.read()

def _read_json_bytes(blob: bytes) -> list[dict]:
    obj = json.loads(blob.decode('utf-8-sig'))
    if not isinstance(obj, list):
        raise ValueError('Prompt JSON must be a list of prompt records')
    return obj

def _read_from_zip(blob: bytes) -> list[dict]:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        matches = [p for p in zf.namelist() if p.endswith('/01_Acne/json/acne_all_300.json')]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one master JSON inside ZIP, found {len(matches)}')
        return _read_json_bytes(zf.read(matches[0]))

def _github_archive(url: str) -> list[dict]:
    # Only public github.com/<owner>/<repo> repository URLs are supported.
    m = re.fullmatch(r'https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?', url)
    if not m:
        raise ValueError('Use a public GitHub repository root URL, or the raw master JSON URL')
    owner, repo = m.group(1),m.group(2)
    errors=[]
    for branch in ('main','master'):
        u=f'https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip'
        try:
            return _read_from_zip(_read_url(u))
        except Exception as e:
            errors.append(f'{branch}: {type(e).__name__}: {e}')
    raise RuntimeError('Could not retrieve dataset from public repository: '+'; '.join(errors))

def load_prompts(source: str | Path) -> list[dict]:
    """Load and validate the full 300-record dataset from a local path or URL.

    Use positive_prompt and negative_prompt verbatim (never reconstruct them).
    """
    source=str(source)
    if source.startswith(('https://','http://')):
        if re.fullmatch(r'https?://(?:www\.)?github\.com/[\w.-]+/[\w.-]+(?:\.git)?/?',source):
            rows=_github_archive(source)
        elif source.lower().endswith('.zip'):
            rows=_read_from_zip(_read_url(source))
        elif source.lower().endswith('.json'):
            # github.com/.../blob/... JSON URLs are web pages; use raw.githubusercontent.com.
            if '/blob/' in source:
                raise ValueError('Use the raw.githubusercontent.com URL, not a GitHub blob page')
            rows=_read_json_bytes(_read_url(source))
        else:
            raise ValueError('Provide a public GitHub repo root URL, raw JSON URL, or ZIP URL')
    else:
        p=Path(source)
        if p.is_dir():
            matches = list(p.rglob('01_Acne/json/acne_all_300.json'))
            if len(matches) == 0:
                matches=list(p.rglob('acne_all_300.json'))
            if len(matches)!=1:
                raise ValueError(f'Expected one acne_all_300.json under {p}, got {len(matches)}')
            rows=_read_json_bytes(matches[0].read_bytes())
        elif p.suffix.lower()=='.zip':
            rows=_read_from_zip(p.read_bytes())
        elif p.suffix.lower()=='.json':
            rows=_read_json_bytes(p.read_bytes())
        else:
            raise ValueError('Provide the dataset directory, .zip, or master .json path')
    verify_prompts(rows)
    return rows

def verify_prompts(rows: list[dict]) -> None:
    if len(rows)!=300 or len({x.get('prompt_id') for x in rows})!=300:
        raise ValueError('Expected 300 records with distinct IDs')
    ids={f'ACNE_{i:03d}' for i in range(1,301)}
    if {x['prompt_id'] for x in rows} != ids:
        raise ValueError('Incorrect acne prompt IDs')
    if Counter(x.get('split') for x in rows) != {'train':210,'val':45,'test':45}:
        raise ValueError('Split counts differ from source dataset')
    if len({x['positive_prompt'] for x in rows}) != 300:
        raise ValueError('Duplicate positive prompt strings')
    groups=defaultdict(list)
    for rec in rows:
        if not isinstance(rec.get('positive_prompt'),str) or not rec['positive_prompt'].strip():
            raise ValueError(f'Missing positive prompt: {rec.get("prompt_id")}')
        if not isinstance(rec.get('negative_prompt'),str) or not rec['negative_prompt'].strip():
            raise ValueError(f'Missing negative prompt: {rec.get("prompt_id")}')
        groups[rec['prompt_group_id']].append(rec)
    if len(groups)!=60 or any(len(g)!=5 or len({x['split'] for x in g})!=1 for g in groups.values()):
        raise ValueError('Invalid grouped split — would risk prompt leakage')

def iter_generation_jobs(rows, split: str='all'):
    """Yield literal prompt/negative and five deterministic seeds; no image work."""
    if split not in ('all','train','val','test'):
        raise ValueError('split must be one of all/train/val/test')
    for rec in rows:
        if split!='all' and rec['split']!=split:
            continue
        for index in range(int(rec['images_per_prompt'])):
            yield {
                'prompt_id': rec['prompt_id'],
                'prompt_group_id': rec['prompt_group_id'],
                'split': rec['split'],
                'phenotype': rec['phenotype'],
                'positive_prompt': rec['positive_prompt'],
                'negative_prompt': rec['negative_prompt'],
                'image_index': index,
                'seed': int(rec['seed_base']) + index,
                'output_stem': f"{rec['prompt_id']}_seed{int(rec['seed_base'])+index}",
            }

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,help='Local dataset directory, ZIP, master JSON or public GitHub URL')
    parser.add_argument('--split',default='all',choices=('all','train','val','test'))
    parser.add_argument('--limit',type=int,default=3)
    args=parser.parse_args()
    recs=load_prompts(args.source)
    from itertools import islice
    print('Verified 300 exact prompt records; 1500 generation jobs (all splits).')
    for job in islice(iter_generation_jobs(recs,args.split),args.limit):
        print(json.dumps(job,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
