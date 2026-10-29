"""Run on Kaggle AFTER loading the exact SD2.1 tokenizer. No truncation."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def validate(records,tokenizer,max_content_tokens=150):
    # CLIP uses bos/eos; count text tokens WITHOUT special tokens.
    results=[];over=[]
    for r in records:
        positive=tokenizer(r['positive_prompt'],add_special_tokens=False,truncation=False)['input_ids']
        negative=tokenizer(r['negative_prompt'],add_special_tokens=False,truncation=False)['input_ids']
        n=len(positive);m=len(negative)
        expected_chunks=max(1,(n+74)//75)
        rec={"prompt_id":r['prompt_id'],"content_tokens":n,"negative_content_tokens":m,"chunks":expected_chunks,"conditioning_positions":77*expected_chunks,"over_limit":n>max_content_tokens}
        results.append(rec)
        if n>max_content_tokens:over.append(rec)
    if over:
        print('FAIL: overlength prompts:',over[:10])
        raise ValueError(f'{len(over)} overlength prompts; fix before generation; no silent truncation')
    return results

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='04_Hyperpigmentation/json/hyperpigmentation_all_300.json')
    parser.add_argument('--tokenizer',default='sd2-community/stable-diffusion-2-1-base',help='Model path or HF repo containing tokenizer')
    parser.add_argument('--output',default='hyperpigmentation_token_report.csv')
    args=parser.parse_args()
    from transformers import CLIPTokenizer
    tok=CLIPTokenizer.from_pretrained(args.tokenizer,subfolder='tokenizer')
    rows=json.loads(Path(args.input).read_text(encoding='utf-8'))
    report=validate(rows,tok)
    with open(args.output,'w',newline='',encoding='utf-8') as fw:
        writer=csv.DictWriter(fw,fieldnames=list(report[0]))
        writer.writeheader();writer.writerows(report)
    print('PASSED',len(report),'max tokens=',max(x['content_tokens'] for x in report),'report=',args.output)

if __name__=='__main__':main()
