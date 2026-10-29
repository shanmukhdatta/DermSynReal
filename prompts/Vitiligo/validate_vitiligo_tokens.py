"""Run in Kaggle after installing transformers; validates exact SD 2.1 CLIP content tokens."""
import argparse, json, pathlib

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--json',default='03_Vitiligo/json/vitiligo_all_300.json')
    p.add_argument('--tokenizer',default='sd2-community/stable-diffusion-2-1-base',help='Must match inference base model')
    p.add_argument('--max-content-tokens',type=int,default=150)
    args=p.parse_args()
    from transformers import CLIPTokenizer
    tokenizer=CLIPTokenizer.from_pretrained(args.tokenizer,subfolder='tokenizer')
    data=json.loads(pathlib.Path(args.json).read_text(encoding='utf-8'))
    limit=tokenizer.model_max_length
    if limit!=77: raise ValueError(f'Unexpected CLIP window: {limit}')
    failed=[]; counts=[]
    for rec in data:
        content_ids=tokenizer(rec['positive_prompt'],add_special_tokens=False,truncation=False)['input_ids']
        n=len(content_ids);counts.append(n)
        rec['sd21_content_tokens']=n
        rec['token_validation_status']='PASSED' if n<=args.max_content_tokens else 'TOO_LONG'
        if n>args.max_content_tokens:failed.append((rec['prompt_id'],n))
    output=pathlib.Path(args.json).with_name('vitiligo_all_300_token_checked.json')
    output.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'Checked {len(counts)} prompts: min={min(counts)} max={max(counts)}, exceeding={len(failed)}')
    print('Saved:',output)
    if failed:
        print('First failures:',failed[:30]);raise SystemExit('ERROR: prompt budget exceeded. Edit and revalidate, do not truncate.')

if __name__=='__main__':main()
