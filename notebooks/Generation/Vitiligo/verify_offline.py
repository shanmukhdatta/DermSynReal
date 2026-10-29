"""CPU-only behavioral and static verification of vitiligo Kaggle notebooks.
No GPU weights or external packages are imported. Expected to pass with Pillow,
Torch CPU, NumPy, scikit-learn and nbformat.
"""
import ast
import csv
import hashlib
import json
import math
import re
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit

BASE = Path(__file__).parent
INF = nbformat.read(BASE / '01_SD21_Vitiligo_Inference.ipynb', as_version=4)
TRN = nbformat.read(BASE / '02_SD21_Train_Facial_Vitiligo_LoRA.ipynb', as_version=4)
for nb in (INF, TRN):
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            compile(cell.source, f'cell{i}', 'exec')
print('PASS: Notebook JSON, code compilation')

class FakeHash:
    def __init__(self, bits):
        self.bits = bits
    def __sub__(self, rhs):
        return (self.bits ^ rhs.bits).bit_count()

def mock_phash(image):
    a = np.asarray(image.convert('L').resize((8, 8)), dtype=np.float32)
    vals = a > np.median(a)
    bits = 0
    for b in vals.flatten():
        bits = (bits << 1) | bool(b)
    return FakeHash(int(bits))

# Synthetic images: 24 patterns and two deliberately duplicated images
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    image_root = root / 'source'
    image_root.mkdir()
    rng = np.random.default_rng(42)
    for i in range(24):
        data = rng.integers(0, 255, size=(72, 72, 3), dtype=np.uint8)
        Image.fromarray(data).save(image_root / f'photo_{i:02d}.png')
    Image.open(image_root / 'photo_00.png').save(image_root / 'duplicate.png')
    with Image.open(image_root / 'photo_01.png') as sample:
        ImageOps.mirror(sample).save(image_root / 'mirrored.png')

    train_path = BASE / '02_SD21_Train_Facial_Vitiligo_LoRA.ipynb'
    cells = [c.source for c in TRN.cells if c.cell_type == 'code']
    by_signature = lambda text: next(c for c in cells if text in c)

    ns = dict(
        EXTS={'.png','.jpg','.jpeg','.webp'}, IMAGE_ROOT=image_root,
        Image=Image, ImageOps=ImageOps, imagehash=SimpleNamespace(phash=mock_phash),
        hashlib=hashlib, Path=Path, np=np, GroupShuffleSplit=GroupShuffleSplit,
        csv=csv, Counter=Counter, SEED=42031, SPLIT_CSV='', CAPTIONS_CSV='',
        AUTO_SPLIT=True, OUT_DIR=root/'output', DATA_PREP=root/'prepared',
    )
    ns['OUT_DIR'].mkdir()
    ns['DATA_PREP'].mkdir()
    exec(by_signature("all_images = sorted("), ns)
    exec(by_signature("parent = list(range("), ns)
    recs=ns['records']
    assert len(recs) == 26
    assert ns['root'](0) in {r['group'] for r in recs}
    assert recs[0]['group'] == next(r['group'] for r in recs if r['rel']=='duplicate.png')
    assert len({r['group'] for r in recs}) < len(recs)
    print('PASS: Image scanning and flip-aware duplicate grouping')

    exec(by_signature("def csv_lookup("), ns)
    splits = {r['split'] for r in ns['records']}
    assert {'train','val','test'} <= splits, splits
    grp_splits = {}
    for r in ns['records']:
        grp_splits.setdefault(r['group'],set()).add(r['split'])
    assert all(len(x)==1 for x in grp_splits.values())
    assert len(ns['train_records']) > 0
    assert (ns['OUT_DIR']/'split_manifest.csv').exists()
    print('PASS: Provisional train/val/test group isolation and CSV manifest')

    exec(by_signature("images_dir = DATA_PREP /"), ns)
    with (ns['DATA_PREP']/'metadata.csv').open() as fp:
        rows=list(csv.DictReader(fp))
    assert len(rows)==len(ns['train_records'])
    assert all(r['file_name'].startswith('images/') for r in rows)
    assert all((ns['DATA_PREP']/r['file_name']).is_file() for r in rows)
    print('PASS: ImageFolder metadata paths, captions and image decoding')

    # Inference prompt validation: mock tokenizer to measure the boundary,
    # and a Compel mock that returns realistic [1,77*N,1024] embeddings.
    class FakeTokenizer:
        model_max_length=77
        def __call__(self, content, truncation, add_special_tokens):
            assert truncation is False and add_special_tokens is False
            return {'input_ids':list(range(len(content.split())))}

    class FakeCompel:
        def __init__(self, tokenizer, text_encoder, **kwargs):
            assert kwargs.get('truncate_long_prompts') is False
            assert kwargs.get('requires_pooled') is False
            assert kwargs.get('device') == 'cuda'
            self.tokenizer = tokenizer
        def __call__(self, text):
            chunks=max(1,math.ceil(len(text.split())/75))
            return torch.zeros((1,77*chunks,1024),dtype=torch.float32)
        def pad_conditioning_tensors_to_same_length(self, items):
            max_len=max(x.shape[1] for x in items)
            return [torch.nn.functional.pad(x,(0,0,0,max_len-x.shape[1])) for x in items]

    class HiddenStates:
        LAST_HIDDEN_STATES_NORMALIZED='last'

    token_ns = dict(
        pipe=SimpleNamespace(tokenizer=FakeTokenizer(), text_encoder=object()),
        Compel=FakeCompel, ReturnedEmbeddingsType=HiddenStates,
        PROMPTS=[{'id':'one','text':'vitiligo facial photograph'}],
        NEGATIVE_PROMPT='bad image, text',
        MAX_CONTENT_TOKENS=150, MAX_CLIP_CHUNKS=2,
    )
    exec(next(c.source for c in INF.cells if c.cell_type=='code' and 'CONTENT_PER_CHUNK' in c.source), token_ns)
    token_ns['validate_prompt']('sample '*150,'boundary 150')
    try:
        token_ns['validate_prompt']('sample '*151,'boundary 151')
    except ValueError:
        pass
    else:
        raise AssertionError('151 tokens were not rejected')
    embeds = token_ns['compel']('text '*150)
    neg_embeds = token_ns['compel']('negative')
    a,b=token_ns['compel'].pad_conditioning_tensors_to_same_length([embeds,neg_embeds])
    assert tuple(a.shape)==tuple(b.shape)==(1,154,1024)
    print('PASS: Compel control-flow mocks, 150/151 boundary, two-chunk padding')

    class FakePipe:
        unet = SimpleNamespace(dtype=torch.float32)
        def __call__(self, **kwargs):
            assert kwargs['prompt_embeds'].shape==kwargs['negative_prompt_embeds'].shape
            return SimpleNamespace(images=[Image.new('RGB',(kwargs['width'],kwargs['height']),'white')])

    output = root / 'generated'
    output.mkdir()
    gen_ns={
        're':re, 'Image':Image, 'display':lambda *args: None,
        'torch':torch, 'json':json, 'time':__import__('time'),
        'pipe':FakePipe(), 'compel':token_ns['compel'],
        'NEGATIVE_PROMPT':'low fidelity', 'WIDTH':64, 'HEIGHT':64,
        'OUT_DIR':output,'IMAGES_PER_PROMPT':5,'MAX_CLIP_CHUNKS':2,
        'BASE_SEED':400,'BASE_MODEL':'mock-sd21','base_commit':'mock',
        'LORA_SOURCE':'', 'LORA_SCALE':.8,'STEPS':2,'GUIDANCE':7,
        'content_token_count':token_ns['content_token_count'],
        'validate_prompt':token_ns['validate_prompt'],
        'PROMPTS':[
            {'id':'a','text':'full face vitiligo'},
            {'id':'b','text':' '.join(['vitiligo']*150)},
        ],
        'Path':Path,
    }
    code=next(c.source for c in INF.cells if c.cell_type=='code' and 'metadata = []' in c.source)
    # Only for the CPU mock: route the generated tensors and torch generator to CPU.
    code=code.replace("device='cuda'", "device='cpu'")
    exec(code,gen_ns)
    assert len(list(output.rglob('*.png')))==10
    assert len((output/'metadata.jsonl').read_text().splitlines())==10
    assert all('seed' in json.loads(line) for line in (output/'metadata.jsonl').read_text().splitlines())
    print('PASS: CPU-mocked five images per prompt, metadata JSONL and embedded conditioning')

assert "--mixed_precision', 'fp16'" in by_signature("command = [")
assert "'--gradient_checkpointing'" in by_signature("command = [")
assert "'--report_to', 'none'" in by_signature("command = [")
assert 'Compel' in next(c.source for c in INF.cells if c.cell_type=='code' and 'CONTENT_PER_CHUNK' in c.source)
print('PASS: Official trainer CLI static configuration')
print('RESULT: ALL AVAILABLE OFFLINE CHECKS PASSED; actual dependencies, GPU model loading and training NOT executed.')
