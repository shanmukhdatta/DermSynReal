"""Read the exact authored prompts from local JSON or a raw public GitHub URL."""
import json
from pathlib import Path
from urllib.request import urlopen
from collections import defaultdict

def load_fungal_prompts(path_or_raw_github_url, *, split=None, exclude_uncertain=False):
    src = str(path_or_raw_github_url)
    if src.startswith("https://"):
        with urlopen(src, timeout=60) as resp:
            content = resp.read().decode("utf-8-sig")
    else:
        content = Path(src).read_text(encoding="utf-8-sig")
    entries = json.loads(content)
    if not isinstance(entries, list) or not entries:
        raise ValueError("Expected a nonempty JSON list of prompt records")
    seen = set()
    by_group = defaultdict(set)
    for item in entries:
        for field in ("prompt_id", "prompt_group_id", "split", "positive_prompt", "negative_prompt", "inference_use_status"):
            if field not in item or item[field] is None:
                raise ValueError("Missing field: " + field)
        pid = item["prompt_id"]
        if pid in seen:
            raise ValueError("Duplicate prompt ID: " + pid)
        seen.add(pid)
        by_group[item["prompt_group_id"]].add(item["split"])
    if any(len(x) > 1 for x in by_group.values()):
        raise ValueError("Prompt group leakage across splits")
    if split is not None and split not in {"train", "val", "test"}:
        raise ValueError("Expected split=train, val, or test")
    if split is not None:
        entries = [x for x in entries if x["split"] == split]
    if exclude_uncertain:
        entries = [x for x in entries if x["inference_use_status"] != "HOLD_FOR_DIAGNOSTIC_REVIEW"]
    return entries
