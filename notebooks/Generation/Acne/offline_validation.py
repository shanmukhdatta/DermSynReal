from pathlib import Path
import nbformat, ast, tempfile, sys, types, random, json, hashlib, csv
import numpy as np
from PIL import Image, ImageOps
root=Path(__file__).resolve().parent
notebooks={p.name:nbformat.read(p,as_version=4) for p in root.glob('*.ipynb')}
for name,nb in notebooks.items():
 nbformat.validate(nb)
 for cell in nb.cells:
  if cell.cell_type=='code':
   ast.parse(cell.source)
   compile(cell.source,filename=name,mode='exec')
 print('PASS notebook syntax and schema:',name)

# CPU-only test of dataset scanning, group splitting and imagefolder output with synthetic mock images.
with tempfile.TemporaryDirectory() as tmp:
 inp=Path(tmp)/'input'; inp.mkdir()
 for i in range(26):
  a=np.random.default_rng(i).integers(0,256,(180,210,3),dtype=np.uint8)
  Image.fromarray(a).save(inp/f'{i:03}.png')
 # exact duplicate in alternate file format
 Image.open(inp/'000.png').save(inp/'duplicate.png')
 dummy=types.ModuleType('imagehash')
 def phash(im):
  arr=np.array(im.resize((8,8)).convert('L')).flatten()
  avg=arr.mean()
  return f'{sum((1 << i) for i,z in enumerate(arr) if z>=avg):016x}'
 dummy.phash=phash
 sys.modules['imagehash']=dummy
 nb=notebooks['02_SD21_Train_Facial_Acne_LoRA.ipynb']
 env={'IMAGE_ROOT':inp,'SPLIT_CSV':'','CAPTIONS_CSV':'','AUTO_SPLIT':True,
      'SEED':34021,'OUT':Path(tmp)/'output','DATA_PREP':Path(tmp)/'prepared',
      'RESOLUTION':512,'TRAIN_WITH_CYSTIC_TI':False,'hashlib':hashlib,'np':np,'csv':csv,'json':json,'Image':Image,'ImageOps':ImageOps,'display':lambda x:None}
 env['OUT'].mkdir();env['DATA_PREP'].mkdir()
 for code_marker in ('import imagehash','def read_lookup','import shutil\n# Clear stale'):
  c=next(c.source for c in nb.cells if c.cell_type=='code' and c.source.startswith(code_marker))
  exec(compile(c,'notebook-cell','exec'),env)
 manifest=list(csv.DictReader(open(env['OUT']/'acne_data_manifest.csv',encoding='utf-8')))
 assert len(manifest)==27
 assert len({r['group_id'] for r in manifest}) == 26
 split_000=next(r['split'] for r in manifest if r['filename']=='000.png')
 assert next(r['split'] for r in manifest if r['filename']=='duplicate.png')==split_000
 train_count=sum(r['split']=='train' for r in manifest)
 lines=[json.loads(x) for x in open(env['DATA_PREP']/'metadata.jsonl',encoding='utf-8')]
 assert len(lines)==len({r['group_id'] for r in manifest if r['split']=='train'})>0
 assert len(list(env['DATA_PREP'].glob('image_*.jpg')))==len(lines)
 assert all((env['DATA_PREP']/r['file_name']).exists() for r in lines)
 print(f'PASS CPU preprocessing mock: {len(manifest)} images, 26 groups, {train_count} train images, metadata valid')

# Verify 5 images/prompt configuration and prompt names are unique, by extracting the two pure config cells.
inf=notebooks['01_SD21_Acne_Inference.ipynb']
cfg=next(c.source for c in inf.cells if c.cell_type=='code' and c.source.startswith('BASE_MODEL ='))
prompts=next(c.source for c in inf.cells if c.cell_type=='code' and c.source.startswith('PROMPTS ='))
env={'Path':Path}
exec(cfg,env);exec(prompts,env)
assert env['IMAGES_PER_PROMPT']==5
assert len(env['PROMPTS'])==3
assert len({p['id'] for p in env['PROMPTS']})==3
print('PASS generation plan:',len(env['PROMPTS'])*env['IMAGES_PER_PROMPT'],'outputs for sample prompts')
print('All offline checks passed; no CUDA/model execution was attempted.')

# Static preflight tests for critical external trainer interactions (actual HF CUDA run requires Kaggle).
train=notebooks['02_SD21_Train_Facial_Acne_LoRA.ipynb']
cells='\n'.join(c.source for c in train.cells if c.cell_type=='code')
assert '\"tensorboard==2.20.0\"' in cells, 'Missing tensorboard dependency'
assert "'--mixed_precision','fp16'" in cells, 'Official trainer also requires explicit mixed precision flag'
assert "'--image_column','image'" in cells and "'--caption_column','text'" in cells
assert "USE_8BIT_ADAM = False" in cells, 'Unexpected optimizer default'
assert "'--use_8bit_adam'" in cells and 'bitsandbytes' in cells, 'Optional optimizer dependency guard missing'
assert 'TRAIN_WITH_CYSTIC_TI = False' in cells
print('PASS static trainer preflight: mixed precision, dataset columns, optimizer dependencies, TI safeguard')
