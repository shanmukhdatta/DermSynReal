"""Exact SD 2.1 CLIP token check. Run on Kaggle with Transformers installed."""
import argparse
import csv
import json
from pathlib import Path

def validate(records, tokenizer, max_tokens=150):
    result=[]; bad=[]
    for item in records:
        pos=tokenizer(item['positive_prompt'],add_special_tokens=False,truncation=False)['input_ids']
        neg=tokenizer(item['negative_prompt'],add_special_tokens=False,truncation=False)['input_ids']
        n,m=len(pos),len(neg)
        rec={'prompt_id':item['prompt_id'],'positive_content_tokens':n,'negative_content_tokens':m,
             'positive_chunks':max(1,(n+74)//75),'positive_conditioning_positions':77*max(1,(n+74)//75),
             'over_limit':n>max_tokens,'negative_over_limit':m>max_tokens}
        result.append(rec)
        if n>max_tokens or m>max_tokens:bad.append(rec)
    if bad:
        raise ValueError(f'{len(bad)} prompts exceed content-token cap; sample: {bad[:5]}')
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='06_Normal_Skin/json/normal_skin_all_300.json')
    parser.add_argument('--tokenizer',default='sd2-community/stable-diffusion-2-1-base')
    parser.add_argument('--output',default='normal_skin_token_report.csv')
    args=parser.parse_args()
    from transformers import CLIPTokenizer
    tok=CLIPTokenizer.from_pretrained(args.tokenizer,subfolder='tokenizer')
    records=json.loads(Path(args.input).read_text(encoding='utf-8'))
    report=validate(records,tok)
    with open(args.output,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(report[0].keys()));writer.writeheader();writer.writerows(report)
    print('PASS',len(report),'records; max positive tokens:',max(x['positive_content_tokens'] for x in report))

if __name__=='__main__': main()
