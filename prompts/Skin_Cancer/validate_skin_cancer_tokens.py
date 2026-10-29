"""Kaggle: check the exact SD2.1 pipeline tokenizer before ANY image generation.
Run in an environment with `diffusers` / `transformers` installed; uses pipe.tokenizer.
The 150-token content cap includes all positive prompt textual tokens excluding
CLIP BOS/EOS and other special tokens. TI tokenizer additions are supported.
"""
import json
from pathlib import Path

MAX_CONTENT_TOKENS=150

def check_prompts(records, tokenizer):
    valid=[]; failures=[]
    for r in records:
        ids=tokenizer(r['positive_prompt'], add_special_tokens=False,
                      truncation=False)['input_ids']
        content_n=len(ids)
        max_content_per_chunk=tokenizer.model_max_length-2
        if max_content_per_chunk!=75:
            raise RuntimeError(f'Expected SD2.1 75 text tokens per chunk; got {max_content_per_chunk}')
        n_chunks=max(1,(content_n+max_content_per_chunk-1)//max_content_per_chunk)
        r['sd21_content_tokens']=content_n
        r['token_validation_status']='PASS' if content_n<=MAX_CONTENT_TOKENS and n_chunks<=2 else 'FAIL'
        r['conditioning_chunks']=n_chunks
        (valid if r['token_validation_status']=='PASS' else failures).append(r)
    return valid,failures

def main(json_path, pipe):
    rows=json.loads(Path(json_path).read_text(encoding='utf-8'))
    good,bad=check_prompts(rows,pipe.tokenizer)
    print(f'PASS {len(good)} / FAIL {len(bad)}')
    if bad:
        for r in bad[:15]: print('OVERLENGTH:',r['prompt_id'],r['sd21_content_tokens'])
        raise ValueError('STOP: prompt exceeds 150-token cap; edit and recheck before generation')
    return good

# Usage AFTER loading pipe and optional token additions:
# good_rows=main('05_Skin_Cancer/json/skin_cancer_all_300.json', pipe)
# Then encode each row['positive_prompt'] via Compel; do not reconstruct it.
