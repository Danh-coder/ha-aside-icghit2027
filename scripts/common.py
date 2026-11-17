"""Publication paths and deterministic I/O, shared by command-line entry points."""
from pathlib import Path
import gzip,hashlib,json,sys,urllib.request,shutil,os

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ha-aside-paper'))

def longpath(path):
    path=Path(path).resolve()
    if os.name=='nt' and not str(path).startswith('\\\\?\\'):
        return Path('\\\\?\\'+str(path))
    return path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def lines(path):
    path=Path(path);opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf8') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def download(url,target,expected,cache=None):
    target=longpath(target);cache=longpath(cache) if cache else None
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(target)!=expected:raise ValueError(f'Modified file: {target}')
        return
    part=target.with_name(target.name+'.partial')
    if cache and Path(cache).is_file():
        if sha(cache)!=expected:raise ValueError(f'Cache checksum mismatch: {cache}')
        shutil.copyfile(cache,part)
    else:
        request=urllib.request.Request(url,headers={'User-Agent':'HA-ASIDE-ICGHIT2027-reproduction'})
        with urllib.request.urlopen(request,timeout=120) as response,part.open('wb') as f:
            shutil.copyfileobj(response,f)
    if sha(part)!=expected:raise ValueError(f'Download checksum mismatch: {url}')
    part.replace(target)

def decode_prefix(tokenizer,ids):
    stop=next((i for i,t in enumerate(ids) if t in {151643,151645}),len(ids))
    return tokenizer.decode(ids[:stop],skip_special_tokens=True)
