"""Validate DermaLens Acne prompts with the ACTUAL SD 2.1 tokenizer (no silent truncation).

Run in a Kaggle notebook AFTER creating its `pipe` object:

    from pathlib import Path
    import sys
    root = Path('/kaggle/input/YOUR_PROMPT_DATASET/DermaLens_SD21_Prompts')
    sys.path.insert(0, str(root))
    from validate_acne_prompts_for_kaggle import validate_and_write
    report = validate_and_write(root / '01_Acne', pipe.tokenizer,
                                Path('/kaggle/working/acne_prompts_validated'),
                                max_content_tokens=150)

If any positive/negative prompt fails, a readable error file is written and the
function RAISES instead of silently dropping, shortening, or truncating a prompt.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict
import csv, math, json

def validate_and_write(source_dir, tokenizer, output_dir, max_content_tokens=150):
    source_dir=Path(source_dir)
    output_dir=Path(output_dir)
    files=sorted(source_dir.glob('acne_[0-9]*_[0-9]*.csv'))
    if len(files)!=3:
        raise ValueError(f'Expected exactly 3 acne CSV files, found {len(files)} in {source_dir}')
    if int(tokenizer.model_max_length)!=77:
        raise ValueError(f'Expected SD2.1 tokenizer model_max_length=77, got {tokenizer.model_max_length}')
    rows=[]
    for file in files:
        with file.open('r',encoding='utf-8-sig',newline='') as f:
            segment=list(csv.DictReader(f))
        if len(segment)!=100:
            raise ValueError(f'{file} expected 100 records, got {len(segment)}')
        rows.extend(segment)
    if len(rows)!=300 or len(set(r['prompt_id'] for r in rows))!=300:
        raise ValueError('Need exactly 300 unique prompt IDs')

    errors=[]
    bygroup=defaultdict(list)
    for row in rows:
        bygroup[row['prompt_group_id']].append(row)
        for column in ('positive_prompt','negative_prompt'):
            prompt=row[column]
            # Compel uses its own prompt parser to create chunks; these prompts
            # deliberately contain NO Compel weight syntax or embedding brackets.
            ids=tokenizer(prompt, add_special_tokens=False, truncation=False)['input_ids']
            token_count=len(ids)
            column_prefix='positive' if column=='positive_prompt' else 'negative'
            row[f'{column_prefix}_content_tokens']=str(token_count)
            row[f'{column_prefix}_clip_chunks']=str(max(1,math.ceil(token_count / 75)))
            if token_count>max_content_tokens:
                errors.append({'prompt_id':row['prompt_id'],'column':column,
                               'tokens':token_count,'limit':max_content_tokens})
        row['sd21_content_tokens']=row['positive_content_tokens']
        row['token_validation_status']='PASS' if all(
            int(row[f'{c}_content_tokens'])<=max_content_tokens
            for c in ('positive','negative')) else 'FAIL'
        row['expected_conditioning_positions']=str(77*max(
            int(row['positive_clip_chunks']),int(row['negative_clip_chunks'])))
    for group, sub in bygroup.items():
        if len(sub)!=5 or len(set(r['split'] for r in sub))!=1:
            errors.append({'prompt_group_id':group,'error':'cross-split leakage or wrong group size'})

    output_dir.mkdir(parents=True, exist_ok=True)
    report={
       'total':len(rows),
       'model_max_length':int(tokenizer.model_max_length),
       'content_token_limit':max_content_tokens,
       'pass_count':sum(r['token_validation_status']=='PASS' for r in rows),
       'failures':errors,
       'positive_token_min':min(int(r['positive_content_tokens']) for r in rows),
       'positive_token_max':max(int(r['positive_content_tokens']) for r in rows),
       'positive_chunk_histogram':dict(Counter(int(r['positive_clip_chunks']) for r in rows)),
       'negative_chunk_histogram':dict(Counter(int(r['negative_clip_chunks']) for r in rows)),
       'split_counts':dict(Counter(r['split'] for r in rows)),
    }
    (output_dir/'tokenizer_validation_report.json').write_text(json.dumps(report,indent=2))
    if errors:
        raise ValueError(f'{len(errors)} prompts fail validation. Inspect {output_dir / "tokenizer_validation_report.json"}; edit source prompts manually. No input was truncated.')
    output_fields=list(rows[0].keys())
    for part_index,source_file in enumerate(files):
        group=rows[part_index*100:(part_index+1)*100]
        with (output_dir/source_file.name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=output_fields)
            writer.writeheader();writer.writerows(group)
    return report
