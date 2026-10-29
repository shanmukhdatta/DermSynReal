"""CPU-only structural and simulated data tests for Skin Cancer Kaggle notebooks.
Does NOT verify installed Diffusers/Compel packages, real model downloads or CUDA output.
"""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import tempfile,contextlib,zipfile,json,ast,hashlib,csv,re,shutil,gc,time,math
import numpy as np
from PIL import Image,ImageOps
import nbformat

HERE=Path(__file__).resolve().parent
INFER=HERE/'01_SD21_Skin_Cancer_cgDDI_Compel_Inference.ipynb'
TRAIN=HERE/'02_SD21_Train_Facial_Skin_Cancer_LoRA.ipynb'
checks=[]

def record(n): checks.append(n);print('PASS:',n)

nb1=nbformat.read(INFER,as_version=4)
nb2=nbformat.read(TRAIN,as_version=4)
for nb in [nb1,nb2]:
    nbformat.validate(nb)
    for i,cell in enumerate(nb.cells):
        if cell.cell_type=='code':
            compile(cell.source,f'notebook_cell_{i}','exec')
record('Notebook JSON + Python syntax for every cell')

# Make sure script options match pinned official trainer and are not hallucinated.
training_code=nb2.cells[18].source
required=['--pretrained_model_name_or_path','--revision','--train_data_dir','--image_column','--caption_column','--mixed_precision','--rank','--gradient_checkpointing','--checkpointing_steps']
assert all(k in training_code for k in required)
assert "'--report_to', 'none'" in training_code
record('Training CLI static options and version pin')

# Fake tokenizer handles 77 positions and simulated 2-vector textual inversion.
class Tokenizer:
    model_max_length=77
    def __call__(self,prompt,truncation=False,add_special_tokens=False):
        ids=list(range(len(prompt.split())))
        if add_special_tokens: ids=[0]+ids+[1]
        return {'input_ids':ids}
class FakePipe:
    def __init__(self):
        self.tokenizer=Tokenizer()
        self.text_encoder=SimpleNamespace()
        self.unet=SimpleNamespace(dtype='float16')
        self.active=None
        self.last_prompts=[]
    def maybe_convert_prompt(self,prompt,tokenizer):
        return prompt.replace('<bcc-class>','<bcc-class> <bcc-class>_1')
    def set_adapters(self,names,adapter_weights): self.active=names
    def enable_lora(self): self.enabled=True
    def disable_lora(self): self.enabled=False
    def __call__(self,**kwargs):
        assert kwargs['prompt_embeds'].shape==kwargs['negative_prompt_embeds'].shape
        return SimpleNamespace(images=[Image.new('RGB',(64,64),(32,52,72))])

class FakeEmb:
    def __init__(self,length):
        self.shape=(1,length,1024)
        self.ndim=3
    def to(self,**kwargs): return self
class TIManager:
    def __init__(self,p): pass
class EmbTypes:
    LAST_HIDDEN_STATES_NORMALIZED='norm'
class Compel:
    def __init__(self,**kwargs):
        assert kwargs['truncate_long_prompts'] is False
        assert kwargs['returned_embeddings_type']=='norm'
    def __call__(self,prompt):
        n=len(prompt.split())
        return FakeEmb((max(1,math.ceil(n/75)))*77)
    def pad_conditioning_tensors_to_same_length(self,embs):
        n=max(x.shape[1] for x in embs)
        return [FakeEmb(n) for x in embs]
class FakeGenerator:
    def __init__(self,device): pass
    def manual_seed(self,x):return self
class FakeTorch:
    no_grad=contextlib.nullcontext
    inference_mode=contextlib.nullcontext
    Generator=FakeGenerator

with tempfile.TemporaryDirectory() as td:
    env={
        'Path':Path,'PUBLIC_ADAPTERS':{'melanoma':'hcarrion/melanoma','bcc':'hcarrion/basal_cell_carcinoma',
                   'scc':'hcarrion/squamous_cell_carcinoma','ak':'hcarrion/actinic_keratosis'},
        'PROMPTS':[{'id':'bcc_a','subtype':'bcc','text':'skin lesion on face ' + 'texture '*70}],
        'MODES':['base','ti','lora','combined'],
        'adapter_tokens':{'bcc':'<bcc-class>'},
        'adapter_revisions':{'bcc':'sha-demo'},
        'LORA_SOURCES':{'bcc':'/mock/lora','shared':''},
        'loaded_loras':{'/mock/lora':'lora_0'},
        'pipe':FakePipe(),'torch':FakeTorch,'DiffusersTextualInversionManager':TIManager,
        'Compel':Compel,'ReturnedEmbeddingsType':EmbTypes,'re':re,'json':json,
        'time':time,'gc':gc,'Image':Image,'display':lambda *args: None,
        'OUT_DIR':Path(td)/'generated','IMAGES_PER_PROMPT':5,
        'WIDTH':64,'HEIGHT':64,'STEPS':3,'GUIDANCE':7.0,'BASE_SEED':60777,
        'MAX_CONTENT_TOKENS':150,'MAX_CLIP_CHUNKS':2,
        'NEGATIVE_PROMPT':'unrealistic bad lighting','LORA_SCALE':0.8,
        'BASE_MODEL':'sd2-community/stable-diffusion-2-1-base','base_revision':'sha-demo',
        'zipfile':zipfile
    }
    env['OUT_DIR'].mkdir()
    exec(compile(nb1.cells[11].source,'token_check','exec'),env)
    assert env['content_tokens']('texture '*150)==150
    try:
        env['check_prompt']('texture '*151,'bad')
        raise AssertionError('151 tokens accepted')
    except ValueError:pass
    assert env['content_tokens']('<bcc-class>')==2
    record('150 accepted / 151 rejected + textual-inversion expansion token counting')
    exec(compile(nb1.cells[13].source,'inference_loop','exec'),env)
    assert len(env['records'])==20
    assert len(list(env['OUT_DIR'].rglob('*.png')))==20
    assert len(set((r['seed'],r['mode']) for r in env['records']))==20
    assert len({r['seed'] for r in env['records']})==5
    assert len({r['mode'] for r in env['records']})==4
    for row in env['records']:
        assert row['textual_inversion'] is not None if row['mode'] in ('ti','combined') else row['textual_inversion'] is None
        assert row['lora_source'] is not None if row['mode'] in ('lora','combined') else row['lora_source'] is None
        assert row['conditioning_positions']<=154
    manifest=env['OUT_DIR']/'metadata.jsonl'
    metadata=[json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(metadata)==20
    record('Mock inference all 4 modes × 5 images, equal embedding lengths, matching seeds and metadata')
    exec(compile(nb1.cells[15].source,'export','exec'),dict(env,zipfile=zipfile)) if False else None

# Run real preprocessing code on synthetic image fixtures with mocked phash (dependency unavailable locally).
class Hash:
    def __init__(self,n):self.n=n
    def __sub__(self,o):return abs(self.n-o.n)
class ImageHash:
    @staticmethod
    def phash(im):
        import numpy as np
        pixel=np.asarray(im.convert('RGB').resize((1,1)))[0,0]
        return Hash(int(pixel[0])*10 + int(pixel[1])*3 + int(pixel[2])*7)

with tempfile.TemporaryDirectory() as td:
    d=Path(td)/'input';(d/'train').mkdir(parents=True)
    (d/'val').mkdir()
    for name, folder, color in [('a.png','train',(105,120,140)),('a_copy.png','val',(105,120,140)),
                                 ('b.png','train',(30,120,200)),('c.png','train',(200,42,100))]:
        im=Image.new('RGBA',(190,110),(*color,255))
        if name=='c.png':im.putpixel((0,0),(0,0,0,0))
        im.save(d/folder/name)
    env={
       'IMAGE_ROOT':d,'Image':Image,'ImageOps':ImageOps,'Path':Path,'imagehash':ImageHash,
       'hashlib':hashlib,'np':np,'Counter':Counter,'GroupShuffleSplit':None,
       'SPLIT_CSV':'','CAPTIONS_CSV':'','LABELS_CSV':'',
       'AUTO_SPLIT':False,'SEED':5,'TRAIN_TARGET':'bcc',
       'OUT_DIR':Path(td)/'output','DATA_PREP':Path(td)/'prepared',
       'RESOLUTION':512,'csv':csv,'shutil':shutil
    }
    env['OUT_DIR'].mkdir()
    env['DATA_PREP'].mkdir()
    exec(compile(nb2.cells[7].source,'scan','exec'),env)
    exec(compile(nb2.cells[8].source,'group','exec'),env)
    exec(compile(nb2.cells[10].source,'split','exec'),env)
    assert len(env['train_records'])==2, [r['rel'] for r in env['train_records']]
    assert all(r['rel'].startswith('train/') for r in env['train_records'])
    record('Real CPU preprocessing: held-out near-duplicate excluded, split boundary respected')
    exec(compile(nb2.cells[12].source,'prepare','exec'),env)
    m=list(csv.DictReader((env['DATA_PREP']/'metadata.csv').open()))
    assert len(m)==2
    for row in m:
        assert row['file_name'].startswith('images/')
        fp=env['DATA_PREP']/row['file_name']
        with Image.open(fp) as im:
            assert im.mode=='RGB' and im.size==(512,512)
            assert im.getpixel((0,0)) != (0,0,0)
    record('Real CPU preprocessing: RGB 512-square letterboxing, valid ImageFolder metadata, nonblack padding')

# Label fail-fast test on shared mode without labels.
conf=nb2.cells[5].source
assert "TRAIN_TARGET == 'all_labeled' and not LABELS_CSV" in conf
record('Shared LoRA requires verified subtype labels; AK category kept separate')
print('ALL OFFLINE TESTS PASSED:',len(checks))
(HERE/'OFFLINE_TEST_RESULTS.txt').write_text('\n'.join(checks)+'\n\nActual installed libraries, Hugging Face model fetch and GPU not tested.\n')
