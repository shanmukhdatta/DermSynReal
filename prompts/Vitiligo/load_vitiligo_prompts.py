"""Load complete saved facial vitiligo prompts, with no reconstruction or generated text."""
import json
from pathlib import Path

def load_vitiligo_prompts(source):
    source = str(source)
    if source.startswith(('https://','http://')):
        import requests
        response = requests.get(source, timeout=60)
        response.raise_for_status()
        data = response.json()
    else:
        data = json.loads(Path(source).read_text(encoding='utf-8'))
    assert isinstance(data, list) and len(data) == 300, 'Expected exactly 300 records'
    ids = set()
    for rec in data:
        assert rec['prompt_id'] not in ids, 'Duplicate prompt_id'
        ids.add(rec['prompt_id'])
        assert rec['positive_prompt'] and rec['negative_prompt']
        assert rec['split'] in ('train','val','test')
    return data

if __name__ == '__main__':
    import sys
    dataset = load_vitiligo_prompts(sys.argv[1])
    print('Loaded', len(dataset), 'prompts')
    print(dataset[0]['prompt_id'], dataset[0]['positive_prompt'])
