"""Offline smoke checks. No internet/GPU/model weights, so these do NOT prove Kaggle CUDA success."""
import ast, csv, hashlib, json, os, re, sys, tempfile, zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import nbformat
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit

ROOT=Path(__file__).parent
CHECKS=[]
def check(name, fn):
    fn()
    CHECKS.append('PASS '+name)

INFER=nbformat.read(ROOT/'01_SD21_Hyperpigmentation_Compel_Inference.ipynb',as_version=4)
TRAIN=nbformat.read(ROOT/'02_SD21_Train_Facial_Hyperpigmentation_LoRA.ipynb',as_version=4)

def syntax_all():
    for nb in (INFER,TRAIN):
        nbformat.validate(nb)
        for j,c in enumerate(nb.cells):
            if c.cell_type=='code':
                compile(c.source, f'cell {j}', 'exec')
    assert len(INFER.cells)>=16 and len(TRAIN.cells)>=20
check('Both notebook JSONs and all Python cells compile', syntax_all)

def static_model_and_cli():
    s='\n'.join(c.source for c in INFER.cells)
    t='\n'.join(c.source for c in TRAIN.cells)
    assert 'hcarrion/hyperpigmentation' in s and 'hcarrion/solar_lentigo' in s
    assert 'DiffusersTextualInversionManager(pipe)' in s
    assert 'pipe.load_textual_inversion(model_id, revision=revision)' in s
    assert 'pipe.load_lora_weights' in s
    assert 'truncate_long_prompts=False' in s
    assert 'ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED' in s
    assert 'pipe.maybe_convert_prompt' in s
    assert 'pad_conditioning_tensors_to_same_length' in s
    assert 'prompt_embeds=positive_embeds' in s and 'negative_prompt_embeds=negative_embeds' in s
    assert 'MAX_CONTENT_TOKENS = 150' in s and 'MAX_CLIP_CHUNKS = 2' in s
    assert "f'images/{name}'" in t and "'metadata.csv'" in t
    assert 'train_text_to_image_lora.py' in t and '--gradient_checkpointing' in t
    assert "'--mixed_precision', 'fp16'" in t
    assert "'--rank'" in t and "'--report_to', 'none'" in t
    assert "'--use_8bit_adam'" in t
    assert 'for i,caption in enumerate(sample[\'text\'])' in t
check('Adapters, Compel APIs, pinned trainer options and training-caption checks',static_model_and_cli)

# Dynamic actual inference prompt assembly/count checks with a fake tokenizer + pipeline.
class FakeTokenizer:
    model_max_length=77
    def __call__(self, text, **kwargs):
        n=len(text.split())
        # A token here is a mock, not the real SD2.1 tokenizer.
        return {'input_ids':list(range(n))}
class FakePipe:
    tokenizer=FakeTokenizer()
    @staticmethod
    def maybe_convert_prompt(prompt, tokenizer):
        return prompt.replace('<BROAD>', '<BROAD> <BROAD> <BROAD>')  # mock 3-vector embedding

def prompt_boundary():
    ns={'pipe':FakePipe(),'MAX_CONTENT_TOKENS':150,'CONTENT_PER_CHUNK':75,
        'PUBLIC_ADAPTERS':{'broad':'example','solar':'example2'},
        'adapter_tokens':{'broad':'<BROAD>','solar':'<SOLAR>'}}
    cell=INFER.cells[11].source
    tree=ast.parse(cell)
    wanted={'build_prompt','content_tokens','enforce_token_limit'}
    funs=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in wanted]
    exec(compile(ast.Module(body=funs,type_ignores=[]),'<test>','exec'),ns)
    b=ns['build_prompt']({'id':'a','adapter_mode':'broad','text':'neutral face'})
    assert '<BROAD>' in b and ns['content_tokens'](b)==5
    assert ns['enforce_token_limit']('word '*150,'boundary')==150
    try: ns['enforce_token_limit']('word '*151,'over')
    except ValueError: pass
    else: raise AssertionError('151 tokens were not rejected')
    try: ns['enforce_token_limit']('word '*148 + '<BROAD>','expanded')
    except ValueError: pass
    else: raise AssertionError('expanded 151 tokens were not rejected')
    assert '<SOLAR>' in ns['build_prompt']({'adapter_mode':'solar','text':'face'})
    assert ns['build_prompt']({'adapter_mode':'none','text':'face'})=='face'
check('Mock 150/151 token boundaries including textual-inversion expansion',prompt_boundary)

def mock_embed_generation():
    class FakeCompel:
        def __call__(self,s):
            return torch.zeros((1,154 if len(s.split())>75 else 77,1024),dtype=torch.float16)
        @staticmethod
        def pad_conditioning_tensors_to_same_length(arr):
            length=max(x.shape[1] for x in arr)
            return [torch.nn.functional.pad(x,(0,0,0,length-x.shape[1])) for x in arr]
    proc=FakeCompel()
    pos,neg=proc('w '*120),proc('w '*8)
    pos,neg=proc.pad_conditioning_tensors_to_same_length([pos,neg])
    assert pos.shape==neg.shape==(1,154,1024)
    assert torch.isfinite(pos).all() and torch.isfinite(neg).all()
check('Mock two-chunk positive/negative embed equalization',mock_embed_generation)

# Train preprocessing and explicit split leakage prevention, using CPU synthetic image fixtures.
class FakeHash:
    def __init__(self,val): self.val=val
    def __sub__(self,other): return (self.val ^ other.val).bit_count()
    def __repr__(self): return f'{self.val:016x}'
class FakeImageHash:
    @staticmethod
    def phash(img):
        a=np.array(img.convert('L').resize((8,8)),dtype=np.float64).flatten()
        med=np.median(a)
        val=0
        for bit,v in enumerate(a):
            if v>med: val |= (1 << bit)
        return FakeHash(val)

def exercise_split(mode):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)/'root';root.mkdir()
        rng=np.random.default_rng(1234)
        paths=[]
        for i in range(22):
            pixels=rng.integers(0,256,size=(64,64,3),dtype=np.uint8)
            f=root/f'face_{i:02d}.png'
            Image.fromarray(pixels).save(f)
            paths.append(f)
        # Cross-split same pixel content must not enter training.
        same=root/'copy_of_face_00.png';same.write_bytes(paths[0].read_bytes())
        # Need test both explicit and auto split.
        split=root/'split.csv'
        if mode=='csv':
            with split.open('w',newline='',encoding='utf8') as out:
                wr=csv.DictWriter(out,fieldnames=['filename','split']);wr.writeheader()
                for p in paths:
                    wr.writerow({'filename':p.name,'split':'train' if p in paths[:17] else 'val'})
                wr.writerow({'filename':same.name,'split':'test'})
        ns={'Path':Path,'IMAGE_ROOT':root,'EXTS':{'.png'},'Image':Image,'ImageOps':ImageOps,
            'imagehash':FakeImageHash,'hashlib':hashlib,'np':np,'Counter':Counter,
            'GroupShuffleSplit':GroupShuffleSplit,'SEED':52031,'csv':csv,
            'SPLIT_CSV':str(split) if mode=='csv' else '',
            'CAPTIONS_CSV':'','AUTO_SPLIT':True,
            'OUT_DIR':Path(tmp)/'out','DATA_PREP':Path(tmp)/'prepared', 'RESOLUTION':512}
        ns['OUT_DIR'].mkdir();ns['DATA_PREP'].mkdir()
        # This is real code extracted from training notebook, not copied pseudocode.
        for i in (7,8,10):
            exec(compile(TRAIN.cells[i].source,f'train cell {i}','exec'),ns)
        assert ns['train_records']
        if mode=='csv':
            assert not any(r['path']==paths[0] or r['path']==same for r in ns['train_records'])
            assert len(ns['train_records'])<=16
        ns['shutil']=__import__('shutil')
        exec(compile(TRAIN.cells[12].source,'train caption cell','exec'),ns)
        meta=ns['DATA_PREP']/'metadata.csv'
        assert meta.exists()
        with meta.open(newline='',encoding='utf8') as f:
            rows=list(csv.DictReader(f))
        assert rows and len(rows)==len(ns['train_records'])
        for row in rows:
            assert (ns['DATA_PREP']/row['file_name']).is_file()
            assert row['text'] and 'pigment' in row['text']
        assert all(r['group'] not in {x['group'] for x in ns['records'] if x['split']!='train'} for r in ns['train_records'])

def splits():
    exercise_split('csv')
    exercise_split('auto')
check('Real notebook CPU dataset scanning, grouped split exclusion and ImageFolder caption paths',splits)



def actual_loop_mocked_gpu():
    # Execute the exact image-saving notebook cell, substituting only CUDA device strings
    # and network-dependent image-generation objects. This does NOT execute Diffusers.
    class FakeCompel:
        def __call__(self,s):
            return torch.zeros((1, 154 if len(s.split())>75 else 77, 1024),dtype=torch.float16)
        def pad_conditioning_tensors_to_same_length(self, arr):
            longest=max(x.shape[1] for x in arr)
            return [torch.nn.functional.pad(x,(0,0,0,longest-x.shape[1])) for x in arr]
    class FakePipeline:
        unet=SimpleNamespace(dtype=torch.float16)
        tokenizer=SimpleNamespace(model_max_length=77)
        def __call__(self, **kwargs):
            assert kwargs['prompt_embeds'].shape==kwargs['negative_prompt_embeds'].shape
            assert kwargs['prompt_embeds'].device.type=='cpu'
            return SimpleNamespace(images=[Image.new('RGB',(64,64),(123,110,100))])
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp)/'images';out.mkdir()
        prompts=[{'id':'first','subtype':'melasma','adapter_mode':'none'},
                 {'id':'second','subtype':'solar_lentigines','adapter_mode':'solar'}]
        valid={'first':'full face cheek pigmentation '*20,'second':'flat lentigines on cheeks '*20}
        ns={'re':re,'json':json,'time':__import__('time'), 'gc':__import__('gc'),
            'Image':Image,'display':lambda x:None,'torch':torch,
            'PROMPTS':prompts,'validated_prompts':valid,
            'NEGATIVE_PROMPT':'blur, illustration','compel_encoder':FakeCompel(),
            'pipe':FakePipeline(),'MAX_CLIP_CHUNKS':2,'OUT_DIR':out,
            'IMAGES_PER_PROMPT':5,'BASE_SEED':52031,'HEIGHT':64,'WIDTH':64,
            'STEPS':2,'GUIDANCE':4.0,
            'LORA_SOURCE':'','LORA_SCALE':0.8,
            'BASE_MODEL':'fake/base','base_revision':'fake-sha',
            'PUBLIC_ADAPTERS':{'solar':'fake/solar','broad':'fake/broad'},
            'adapter_tokens':{'solar':'<SOLAR>','broad':'<BROAD>'},
            'adapter_revisions':{'solar':'fake-adapter-sha'},
            'content_tokens':lambda s:len(s.split()),
            }
        cell=INFER.cells[13].source
        # For this CPU-only test use CPU allocations in otherwise unchanged generation code.
        cell=cell.replace("device='cuda'", "device='cpu'")
        exec(compile(cell,'inference cell 13 mock GPU','exec'),ns)
        imagefiles=list(out.rglob('*.png'))
        assert len(imagefiles)==10, f'Expected 10 PNGs, saw {len(imagefiles)}'
        manifest=out/'metadata.jsonl'
        assert manifest.exists()
        records=[json.loads(line) for line in manifest.read_text(encoding='utf8').splitlines()]
        assert len(records)==10
        assert len({r['seed'] for r in records})==10
        assert set(r['requested_adapter_mode'] for r in records)=={'none','solar'}
        assert all((out/r['file']).exists() for r in records)
check('Real inference cell with mocked diffusion generates five PNGs per prompt plus valid JSONL',actual_loop_mocked_gpu)


# Extended checks introduced after the independent code review.
def image_preprocessing_alpha_and_aspect():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        rgba=Image.new('RGBA',(160,80),(0,0,0,0))
        # Mark the entire border (and a facial patch in the center): a square crop
        # would lose some of the border, padding should keep it.
        from PIL import ImageDraw
        d=ImageDraw.Draw(rgba)
        d.rectangle((0,0,159,79),outline=(0,100,180,255),width=2)
        d.ellipse((65,20,95,60),fill=(110,60,50,255))
        src=root/'example_rgba.png';rgba.save(src)
        ns={'Image':Image,'ImageOps':ImageOps,'Path':Path,'csv':csv,
            'DATA_PREP':root/'prep','train_records':[{'path':src,'rel':src.name}],
            'RESOLUTION':512,'lookup':lambda m,r:None,'caption_lookup':{}}
        ns['DATA_PREP'].mkdir(); ns['shutil']=__import__('shutil')
        exec(compile(TRAIN.cells[12].source,'preprocess real train cell','exec'),ns)
        out=ns['DATA_PREP']/'images'/'000000.png'
        assert out.exists()
        with Image.open(out) as x:
            assert x.size==(512,512) and x.mode=='RGB'
            arr=np.asarray(x)
            # Corner padding should be neutral grey, never transparent -> black.
            assert arr[0,0].tolist()==[215,215,215]
            assert np.any(arr[:,:,0] == 110)
            assert arr.mean()>5
            assert (arr[128:384,:,2] > 150).any()  # left/right borders preserved
check('Real preprocessing retains rectangular framing and prevents transparent-black output',image_preprocessing_alpha_and_aspect)


def provenance_check_static():
    s='\n'.join(c.source for c in INFER.cells)
    t='\n'.join(c.source for c in TRAIN.cells)
    assert "training_provenance.json" in s and "training_provenance.json" in t
    assert "training_provenance.get('base_model') != BASE_MODEL" in s
    assert "training_provenance.get('base_revision') != base_revision" in s
    assert "'published_textual_inversion_used_during_training': False" in t
    assert "bundle.write(provenance_path, provenance_path.name)" in t
check('Provenance configuration explicitly records TI-versus-LoRA training and checks matching base',provenance_check_static)

if __name__=='__main__':
    for line in CHECKS: print(line)
    print(f'{len(CHECKS)} local checks passed; real Kaggle downloads/CUDA/Compel imports not tested.')
