"""CPU-safe regression for Compel integration. No real external libraries/models needed."""
import ast, math, types
from pathlib import Path
import torch, nbformat

path=Path(__file__).with_name('01_SD21_Acne_Inference.ipynb')
nb=nbformat.read(path,as_version=4)
nbformat.validate(nb)
for idx,c in enumerate(nb.cells):
    if c.cell_type=='code':
        compile(c.source, str(path)+f':cell{idx}', 'exec')

# Fake tokenizer uses space-delimited token ids, plus a simulated multivector TI.
class ToyTokenizer:
    model_max_length=77
    def num_special_tokens_to_add(self, pair=False): return 2
    def encode(self,text,add_special_tokens=False,truncation=False):
        assert truncation is False
        return list(range(len(text.split())))
class ToyPipe:
    tokenizer=ToyTokenizer()
    device=torch.device('cpu')
    class UNet: dtype=torch.float32
    unet=UNet()
    def maybe_convert_prompt(self,text,tokenizer):
        return text.replace('<many-vector>', 'v1 v2 v3')

# Extract functions from real notebook, and execute using real PyTorch in CPU mode.
cell = next(c.source for c in nb.cells if c.cell_type=='code' and 'def encode_conditioning(' in c.source)
tree = ast.parse(cell)
funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
mod=ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[]))
context={'math':math,'torch':torch,'MAX_CONTENT_TOKENS':150,
         'CONTENT_PER_CHUNK':75,'CLIP_WINDOW':77,'MAX_CLIP_CHUNKS':2}
exec(compile(mod,'extracted-notebook-functions','exec'),context)
count=context['count_content_tokens'];check=context['validate_prompt'];encode=context['encode_conditioning']
pipe=ToyPipe()
assert count(' '.join(['a']*150),pipe.tokenizer)==150
assert check(' '.join(['a']*75),'exactly-one-chunk',pipe)['chunks_needed']==1
assert check(' '.join(['a']*76),'just-over-one-chunk',pipe)['chunks_needed']==2
assert check(' '.join(['a']*150),'exactly-two-chunks',pipe)['chunks_needed']==2
for label,text in [('positive-over-limit',' '.join(['a']*151)),('negative-over-limit',' '.join(['bad']*151)),('expanded-token-over-limit',' '.join(['a']*149)+' <many-vector>')]:
    try:check(text,label,pipe)
    except ValueError as e:
        assert 'no truncation' in str(e).lower(), (label,str(e))
    else: raise AssertionError(f'{label} was not rejected')

class ToyEncoder:
    def __init__(self):self.requests=[]
    def build_conditioning_tensor(self,prompt):
        n=math.ceil(len(prompt.split())/75)*77
        self.requests.append(prompt)
        return torch.randn((1,n,1024),dtype=torch.float32)
    def pad_conditioning_tensors_to_same_length(self,tensors):
        n=max(t.shape[1] for t in tensors)
        return [torch.nn.functional.pad(t,(0,0,0,n-t.shape[1])) for t in tensors]
encoder=ToyEncoder()
pos,neg=encode(' '.join(['a']*150),'no blur',pipe,encoder)
assert pos.shape==neg.shape==(1,154,1024)
assert len(encoder.requests)==2
assert torch.isfinite(pos).all() and torch.isfinite(neg).all()
assert 'prompt_embeds=positive_embeds, negative_prompt_embeds=negative_embeds' in '\n'.join(c.source for c in nb.cells)
assert 'prompt=text_prompt, negative_prompt=NEGATIVE_PROMPT' not in '\n'.join(c.source for c in nb.cells)

# Guard against a buggy encoder unexpectedly producing three chunks.
class BadToyEncoder(ToyEncoder):
    def build_conditioning_tensor(self,prompt):
        return torch.randn((1,231,1024),dtype=torch.float32)
try: encode('short','short',pipe,BadToyEncoder())
except ValueError as e:
    assert 'chunks' in str(e)
else: raise AssertionError('Three-chunk embedding was not rejected')

# Re-run preexisting code syntax and training static checks separately.
print('PASS: notebook schema and all code cells compile')
print('PASS: 75/76/150/151 raw-token boundary checks')
print('PASS: textual inversion expansion rejects too-long prompts')
print('PASS: positive/negative equal-length tensor padding [1, 154, 1024]')
print('PASS: >2-chunk embedding rejection and pipeline embeds-only call')
print('LIMITATION: ToyTokenizer/ToyEncoder are mock implementations, not real Compel or SD2.1')
