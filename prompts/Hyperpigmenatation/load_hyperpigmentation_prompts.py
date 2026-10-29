"""Load the COMPLETE stored prompt text; never construct it from metadata."""
import json, sys
from pathlib import Path

def load_prompts(path_or_url):
    url = str(path_or_url)
    if url.startswith(("https://", "http://")):
        import requests
        response=requests.get(url,timeout=60)
        response.raise_for_status()
        records=response.json()
    else:
        records=json.loads(Path(url).read_text(encoding="utf-8"))
    if not isinstance(records,list) or len(records)!=300:
        raise ValueError(f"Expected exactly 300 JSON prompt records; got {len(records) if isinstance(records,list) else type(records)}")
    ids=set()
    for r in records:
        for key in ("prompt_id", "prompt_group_id", "positive_prompt", "negative_prompt", "split", "seed_base"):
            if key not in r or r[key] is None or r[key]=="":raise ValueError(f"Record missing {key}: {r.get('prompt_id')}")
        if r["prompt_id"] in ids:raise ValueError("Duplicate prompt ID")
        ids.add(r["prompt_id"])
    return records

if __name__=="__main__":
    items=load_prompts(sys.argv[1])
    print(f"Loaded {len(items)} complete prompts; first ID: {items[0]['prompt_id']}")
