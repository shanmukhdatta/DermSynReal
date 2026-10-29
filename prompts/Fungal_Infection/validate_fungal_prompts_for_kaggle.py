"""Run this with pipe.tokenizer after loading the exact SD 2.1 pipeline in Kaggle.
It fails closed: output validated JSON only if all prompts respect 150 content tokens.
"""
import json, math
from pathlib import Path

def validate_json(master_json_path, tokenizer, out_json_path, max_content_tokens=150):
    if tokenizer.model_max_length != 77:
        raise ValueError(f"Expected CLIP max_length=77; got {tokenizer.model_max_length}")
    if max_content_tokens != 150:
        raise ValueError("This release is configured for 150 content-token maximum")
    rows = json.loads(Path(master_json_path).read_text(encoding="utf-8"))
    failures = []
    validated = []
    for row in rows:
        new = dict(row)
        for key in ("positive_prompt", "negative_prompt"):
            ids = tokenizer(new[key], add_special_tokens=False, truncation=False)["input_ids"]
            n = len(ids)
            new[key+"_content_tokens"] = n
            if n > max_content_tokens:
                failures.append({"id": new["prompt_id"], "field": key, "tokens": n})
            new[key+"_chunks"] = max(1, math.ceil(n / 75))
        new["sd21_content_tokens"] = new["positive_prompt_content_tokens"]
        new["token_validation_status"] = "EXACT_SD21_TOKENIZER_VALIDATED"
        validated.append(new)
    report = {"rows": len(rows), "positive_max_tokens": max(r["positive_prompt_content_tokens"] for r in validated),
              "negative_max_tokens": max(r["negative_prompt_content_tokens"] for r in validated),
              "failures": failures}
    report_path = Path(out_json_path).with_suffix(".validation.json")
    report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    if failures:
        raise ValueError(f"{len(failures)} prompts too long, see {report_path}. No prompt truncated.")
    Path(out_json_path).write_text(json.dumps(validated, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return report
