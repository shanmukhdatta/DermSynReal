"""Load 300 complete normal facial skin prompts from local JSON or raw GitHub URL."""
import json
import sys
from pathlib import Path

def load_prompts(path_or_url):
    path_or_url=str(path_or_url)
    if path_or_url.startswith(('https://','http://')):
        import requests
        response=requests.get(path_or_url,timeout=60)
        response.raise_for_status()
        records=response.json()
    else:
        records=json.loads(Path(path_or_url).read_text(encoding='utf-8'))
    if not isinstance(records,list) or len(records)!=300:
        raise ValueError('Expected exactly 300 JSON prompt records')
    ids=set()
    for entry in records:
        for key in ['prompt_id','prompt_group_id','positive_prompt','negative_prompt','split','seed_base']:
            if key not in entry or not entry[key]:
                raise ValueError(f'Missing {key} in record {entry.get("prompt_id")}')
        if entry['prompt_id'] in ids: raise ValueError('Duplicate prompt ID')
        ids.add(entry['prompt_id'])
    return records

if __name__=='__main__':
    records=load_prompts(sys.argv[1])
    print('Loaded',len(records),'prompts; first:',records[0]['prompt_id'])
