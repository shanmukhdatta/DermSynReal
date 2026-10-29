"""Download/load complete prompts directly; do not generate or concatenate text."""
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

def load_prompts(path_or_url, split=None):
    source = str(path_or_url)
    if source.startswith(("https://", "http://")):
        req = Request(source, headers={"User-Agent": "DermaLens-Prompt-Loader/1.0"})
        with urlopen(req, timeout=60) as response:
            data = json.load(response)
    else:
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 300:
        raise ValueError("Expected exactly 300 prompt records")
    seen=set()
    for row in data:
        for key in ("prompt_id", "prompt_group_id", "positive_prompt", "negative_prompt", "split", "seed_base"):
            if key not in row or row[key] in (None, ""):
                raise ValueError(f"Missing {key}: {row}")
        if row["prompt_id"] in seen:
            raise ValueError("Duplicate prompt ID")
        seen.add(row["prompt_id"])
    if split is not None:
        if split not in ("train", "val", "test"):
            raise ValueError("Invalid split")
        return [r for r in data if r["split"] == split]
    return data

if __name__ == "__main__":
    rows=load_prompts(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else None)
    print(f"Loaded {len(rows)} full prompts; first ID: {rows[0]['prompt_id']}")
