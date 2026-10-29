"""CPU-only notebook audit: executes original prompt, generation and image-preparation cells with mocks.
This does NOT load real SDXL/Compel or train anything on CUDA.
"""
from pathlib import Path
from types import SimpleNamespace
import tempfile, json, csv, math, hashlib, time, re, shutil, gc, zipfile, sys
import nbformat, numpy as np, torch
from PIL import Image, ImageOps

ROOT=Path(__file__).parent
INFER=nbformat.read(ROOT/'01_SDXL_Normal_Skin_Compel_Inference.ipynb',as_version=4)
TRAIN=nbformat.read(ROOT/'02_SDXL_Train_Normal_Facial_Skin_LoRA.ipynb',as_version=4)
for nb in (INFER,TRAIN):
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type=='code': compile(cell.source,f'cell{i}','exec')
print('PASS — notebooks validated and all code cells compile')

class Tokenizer:
    model_max_length=77
    def __init__(self, extra=0):self.extra=extra
    def __call__(self,s,truncation=False,add_special_tokens=True):
        n=len(s.strip().split()) + (self.extra if s.strip() else 0)
        return {'input_ids':[0]+list(range(1,n+1))+[2]}

class FakeEncoder:
    def __init__(self,pipe):self.pipe=pipe
    def __call__(self,**kw):
        self.pipe.encoded_modes.append(self.pipe.mode)
        return SimpleNamespace(embeds=torch.zeros((1,154,2048)),negative_embeds=torch.zeros((1,154,2048)),
            pooled_embeds=torch.zeros((1,1280)),negative_pooled_embeds=torch.zeros((1,1280)))

class Pipe:
    tokenizer=Tokenizer()
    tokenizer_2=Tokenizer()
    def __init__(self):self.mode=None;self.calls=[];self.encoded_modes=[]
    def disable_lora(self):self.mode='base'
    def enable_lora(self):pass
    def set_adapters(self,names,adapter_weights=None): self.mode='+'.join(names)
    def __call__(self,**kw):
        assert kw['prompt_embeds'].shape==kw['negative_prompt_embeds'].shape
        assert kw['pooled_prompt_embeds'].shape==kw['negative_pooled_prompt_embeds'].shape
        self.calls.append((self.mode, kw['generator'].initial_seed()))
        return SimpleNamespace(images=[Image.new('RGB',(32,32),(120,130,140))])

class PHash:
    def __init__(self,s):self.x=int.from_bytes(hashlib.sha256(s).digest()[:8],'big')
    def __sub__(self,o):return (self.x^o.x).bit_count()
class FakeImageHash:
    @staticmethod
    def phash(im):return PHash(im.tobytes())
sys.modules['imagehash']=FakeImageHash

with tempfile.TemporaryDirectory(prefix='normal_final_') as tmp:
    t=Path(tmp)
    p=Pipe()
    state={'Path':Path,'torch':torch,'re':re,'json':json,'time':time,'gc':gc,'shutil':shutil,
        'Image':Image,'display':lambda x:None,'pipe':p,'START_SEED':456,
        'MODES':['base','texture','custom','combined'], 'PROMPT_MAX_CONTENT_TOKENS':150,
        'IMAGES_PER_PROMPT':5,'STEPS':3,'HEIGHT':64,'WIDTH':64,'GUIDANCE_SCALE':6,
        'OUT_DIR':t/'generations','PROMPTS':[{'id':'normal_example','text':'normal facial skin with pores'}],
        'NEGATIVE_PROMPT':'blurry skin', 'BASE_MODEL':'sdxl','base_sha':'abc',
        'VAE_MODEL':'vae','vae_sha':'def','TEXTURE_LORA':'texture','texture_sha':'ghi',
        'CUSTOM_LORA_SOURCE':'mock_lora','TEXTURE_WEIGHT':0.3,'CUSTOM_WEIGHT':0.7,
        'CompelForSDXL':FakeEncoder,'encoder':FakeEncoder(p)}
    exec(INFER.cells[11].source,state)
    # Boundaries: 150 content tokens accepted by both tokenizers; 151 rejected.
    state['validate_prompt']('word '*150,'exact_150')
    try:state['validate_prompt']('word '*151,'reject_151')
    except ValueError: pass
    else:raise AssertionError('151 content tokens accepted')
    p.tokenizer_2=Tokenizer(extra=1)
    try:state['validate_prompt']('word '*150,'reject_second_tokenizer')
    except ValueError: pass
    else:raise AssertionError('Token count of second tokenizer was ignored')
    p.tokenizer_2=Tokenizer()
    state['torch']=SimpleNamespace(Generator=lambda device:torch.Generator(device='cpu'),inference_mode=torch.inference_mode)
    exec(INFER.cells[13].source,state)
    assert len(state['rows'])==20 and len(list((t/'generations').rglob('*.png')))==20
    assert p.encoded_modes==['base','texture','normal_face','texture+normal_face']
    assert [s for m,s in p.calls[:5]]==[s for m,s in p.calls[5:10]]==[s for m,s in p.calls[10:15]]==[s for m,s in p.calls[15:20]]
    assert len((t/'generations'/'metadata.jsonl').read_text().splitlines())==20
    print('PASS — 150/151 token tests, dual-tokenizer tests, 4 modes, embeddings after adapter selection, matched seeds and 20 fake images')

    data=t/'dataset';(data/'train').mkdir(parents=True);(data/'test').mkdir(parents=True)
    alpha=Image.new('RGBA',(120,80),(255,0,0,0));alpha.putpixel((5,5),(200,100,100,255));alpha.save(data/'train'/'alpha.png')
    Image.new('RGB',(175,95),(130,100,90)).save(data/'train'/'wide.jpg')
    Image.new('RGB',(175,95),(20,30,200)).save(data/'test'/'heldout.jpg')
    env={'Path':Path,'csv':csv,'np':np,'json':json,'Image':Image,'ImageOps':ImageOps,
         'IMAGE_ROOT':data,'SPLIT_CSV':'','CAPTIONS_CSV':'', 'ALLOW_PROVISIONAL_SPLIT':False,
         'DATA_PREP':t/'prep','SEED':1234,'RESOLUTION':128,'LORA_OUT':t/'lorafiles'}
    env['DATA_PREP'].mkdir();env['LORA_OUT'].mkdir()
    exec(TRAIN.cells[7].source,env)
    assert len(env['selected'])==2 and len(env['heldout'])==1 and not env['conflicts']
    exec(TRAIN.cells[9].source,env)
    recs=[json.loads(x) for x in (t/'prep'/'metadata.jsonl').read_text().splitlines()]
    assert len(recs)==2
    for r in recs:
        assert r['file_name'].startswith('images/')
        with Image.open(t/'prep'/r['file_name']) as im:
            assert im.mode=='RGB' and im.size==(128,128) and im.getpixel((0,0)) != (0,0,0)
    print('PASS — real preparation code: train-only split, leakage screen, nonblack transparent/rectangular images, caption metadata')

    # Change held-out to an exact duplicate of the train sample: must reject.
    shutil.copy2(data/'train'/'wide.jpg',data/'test'/'heldout.jpg')
    try:exec(TRAIN.cells[7].source, env)
    except ValueError as exc:
        assert 'collisions' in str(exc)
    else:raise AssertionError('Training did not reject a held-out duplicate')
    print('PASS — held-out exact duplicate triggers explicit stop')

for flag in ['--mixed_precision','--gradient_checkpointing','--use_8bit_adam','--train_data_dir','--rank']:
    assert flag in TRAIN.cells[15].source
assert 'v0.35.1' in TRAIN.cells[13].source
print('PASS — pinned official trainer source reference and CLI flags are present')
print('ALL CPU-OFFLINE AUDIT TESTS PASSED — FULL GPU NOT TESTED')
