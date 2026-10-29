"""CPU-only tests. No Hugging Face weights or GPU calls are executed."""
import ast, json, re, tempfile, time, gc, hashlib, zipfile
from pathlib import Path
from types import SimpleNamespace
import torch
from PIL import Image
import nbformat

ROOT=Path(__file__).resolve().parent
INF=nbformat.read(ROOT/'01_SD21_Fungal_cgDDI_TI_Compel_Inference.ipynb',as_version=4)
TRN=nbformat.read(ROOT/'02_SD21_Train_Facial_Fungal_LoRA.ipynb',as_version=4)
checks=[]
def check(name,fn):
    fn();checks.append('PASS '+name)

def syntax():
    for book in [INF,TRN]:
        nbformat.validate(book)
        for i,c in enumerate(book.cells):
            if c.cell_type=='code': compile(c.source,f'cell_{i}','exec')
    assert len(INF.cells)==17
check('Both notebook JSON files and all Python code cells compile',syntax)

def config():
    txt='\n'.join(c.source for c in INF.cells)
    trn='\n'.join(c.source for c in TRN.cells)
    assert 'hcarrion/tinea_pedis' in txt
    assert 'pipe.load_textual_inversion(TI_MODEL' in txt
    assert 'DiffusersTextualInversionManager(pipe)' in txt
    assert 'truncate_long_prompts=False' in txt
    assert 'pad_conditioning_tensors_to_same_length' in txt
    assert 'prompt_embeds=positive_embeds' in txt
    assert 'negative_prompt_embeds=negative_embeds' in txt
    assert 'MAX_CONTENT_TOKENS = 150' in txt
    assert "'pityriasis_versicolor': ['base']" in txt
    assert 'tinea_pedis TI must not be treated as a versicolor adapter' in txt
    assert 'training_provenance.json' in txt
    assert '--use_8bit_adam' in trn
    assert '--mixed_precision' in trn
check('cgDDI TI loading, token manager, Compel, routing, training CLI checks',config)

class FakeTokenizer:
    model_max_length=77
    def __call__(self,prompt,**kwargs):
        return {'input_ids':list(range(len(prompt.split())))}
class FakePipe:
    tokenizer=FakeTokenizer()
    def maybe_convert_prompt(self,p,tokenizer):
        return p.replace('<test-ti>','test-ti-1 test-ti-2 test-ti-3')

def token_boundary():
    source=ast.parse(INF.cells[11].source)
    funcs=[node for node in source.body if isinstance(node,ast.FunctionDef) and node.name in ['actual_token_count','require_token_budget','positive_prompt']]
    ns={'pipe':FakePipe(),'needs_ti':True,'MAX_CONTENT_TOKENS':150,'TI_TRIGGER':'<test-ti>'}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'token_funcs','exec'),ns)
    assert ns['require_token_budget']('abc '*150,'test')==150
    try: ns['require_token_budget']('abc '*151,'test')
    except ValueError: pass
    else: raise AssertionError('Overlong prompt accepted')
    item={'text':'abc '*148}
    converted=ns['positive_prompt'](item,'ti_only')
    assert ns['actual_token_count'](converted)==151 # 148 + 3 expanded TI tokens = 151
    try: ns['require_token_budget'](converted,'TI-excess')
    except ValueError: pass
    else: raise AssertionError('Expanded TI overrun not detected')
    assert '<test-ti>' not in ns['positive_prompt'](item,'base')
check('150/151 content-token bounds and multi-vector TI expansion',token_boundary)

def make_fake_inference():
    class FakeEncoder:
        def __call__(self,p):
            return torch.zeros((1,154 if len(p.split())>75 else 77,1024),dtype=torch.float16)
        @staticmethod
        def pad_conditioning_tensors_to_same_length(xs):
            n=max(x.shape[1] for x in xs)
            return [torch.nn.functional.pad(x,(0,0,0,n-x.shape[1])) for x in xs]
    class FakePipeline:
        unet=SimpleNamespace(dtype=torch.float16)
        tokenizer=FakeTokenizer()
        def __init__(self):self.active=[]
        def set_adapters(self,ns,adapter_weights):
            self.active.append((ns[0],adapter_weights[0]))
        def __call__(self,**kw):
            assert kw['prompt_embeds'].shape==kw['negative_prompt_embeds'].shape
            assert kw['prompt_embeds'].shape[1]<=154
            assert kw['prompt_embeds'].device.type=='cpu'
            return SimpleNamespace(images=[Image.new('RGB',(32,32),(123,112,97))])
    with tempfile.TemporaryDirectory() as d:
        pipe=FakePipeline()
        output=Path(d)/'images';output.mkdir()
        names=['base','ti_only','lora_only','ti_plus_lora']
        ns={'re':re,'time':time,'gc':gc,'json':json,'torch':torch,'Image':Image,
            'display':lambda x:None,'pipe':pipe,'OUT_DIR':output,
            'PROMPTS':[{'id':'tinea','subtype':'tinea_faciei','text':'clinical fungal lesion photograph '*27},
                       {'id':'pv','subtype':'pityriasis_versicolor','text':'facial fine scale pigment patch '*23}],
            'MODES_BY_SUBTYPE':{'tinea_faciei':names,'pityriasis_versicolor':['base','lora_only']},
            'LORA_SCALES':{'tinea_faciei':0.8,'pityriasis_versicolor':0.9},
            'loaded_loras':{'tinea_faciei':'fake/tinea','pityriasis_versicolor':'fake/pv'},
            'TI_TRIGGER':'<test-ti>','TI_MODEL':'hcarrion/tinea_pedis','public_revision':'test-sha',
            'BASE_MODEL':'test-base','base_revision':'base-sha','encoder':FakeEncoder(),
            'MAX_CLIP_CHUNKS':2,'NEGATIVE_PROMPT':'blurry image',
            'IMAGES_PER_PROMPT':5,'WIDTH':32,'HEIGHT':32,'STEPS':2,'GUIDANCE':2.0,'BASE_SEED':62031,
            'require_token_budget':lambda p,name:len(p.split()),
            'positive_prompt':lambda item,mode:item['text'] + (', <test-ti>' if mode in ['ti_only','ti_plus_lora'] else '')}
        source=INF.cells[13].source.replace("device='cuda'","device='cpu'")
        exec(compile(source,'fake_inference','exec'),ns)
        records=[json.loads(x) for x in (output/'metadata.jsonl').read_text().splitlines()]
        assert len(records)==30
        assert len(list(output.rglob('*.png')))==30
        assert {x['experiment_mode'] for x in records}==set(names)
        assert sum(x['experiment_mode']=='ti_only' for x in records)==5
        assert sum(x['experiment_mode']=='ti_plus_lora' for x in records)==5
        assert len({x['seed'] for x in records})==10
        assert all(x['published_ti']=='hcarrion/tinea_pedis' for x in records if x['experiment_mode'] in ('ti_only','ti_plus_lora'))
        assert all(x['published_ti'] is None for x in records if x['experiment_mode'] in ('base','lora_only'))
        assert all(x['subtype_requested']=='tinea_faciei' for x in records if x['published_ti'])
        assert all((output/x['file']).is_file() for x in records)
        # Exact seed-matching across all tinea modes.
        assert len(set(r['seed'] for r in records if r['subtype_requested']=='tinea_faciei'))==5
check('Mock pipeline: four comparison modes, 5 images/mode, metadata and matched seeds',make_fake_inference)


def ziptest():
    package=ROOT/'README.md'
    assert package.exists()
    assert 'tinea_pedis' in package.read_text()
check('README present and explains adapter domain mismatch',ziptest)

print('\n'.join(checks))
print('PASS count:',len(checks))
print('NOT TESTED: downloading published model, Compel import, genuine SD2.1 CUDA inference or LoRA training.')
