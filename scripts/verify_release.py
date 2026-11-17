"""Verify the publication manifest, staged file scope, and reproduction checks."""
import argparse,hashlib,json,platform,re,subprocess,sys,importlib.metadata
from pathlib import Path
from common import ROOT,read,write,sha

EXCLUDED={'work','outputs','checkpoints','logs','.git','__pycache__','.venv'}
REPORTS={'reproducibility/release_manifest.json','reproducibility/release_verification.json'}

def files():
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if set(rel.parts)&EXCLUDED or rel.parts[:2]==('ha-aside-paper','data'):continue
        if rel.as_posix() in REPORTS or p.suffix in {'.pyc','.partial'}:continue
        yield p

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-manifest',action='store_true')
    args=parser.parse_args()
    inventory={p.relative_to(ROOT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files())}
    manifest=ROOT/'reproducibility/release_manifest.json'
    if args.write_manifest:write(manifest,inventory)
    if not manifest.exists() or read(manifest)!=inventory:raise ValueError('Release file inventory changed')
    secret_patterns=[re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}'),re.compile(r'github_pat_[A-Za-z0-9_]{35,}'),
                     re.compile(r'hf_[A-Za-z0-9]{30,}'),re.compile(r'sk-(?:proj-)?[A-Za-z0-9_-]{30,}'),
                     re.compile(r'^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',re.M)]
    for name,spec in inventory.items():
        p=ROOT/name
        if p.suffix in {'.safetensors','.bin','.pt','.key','.pem'}:raise ValueError(f'Unexpected weight/secret file: {name}')
        if spec['bytes']>95*1024*1024:raise ValueError(f'File exceeds release size limit: {name}')
        if p.suffix in {'.py','.json','.md','.txt','.yml','.cff'}:
            text=p.read_text(encoding='utf8',errors='replace')
            if any(pattern.search(text) for pattern in secret_patterns):raise ValueError(f'Possible credential in {name}')
    if (ROOT/'.git').exists():
        tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
        for name in filter(None,tracked):
            path=Path(name)
            if set(path.parts)&EXCLUDED or path.parts[:2]==('ha-aside-paper','data'):
                raise ValueError(f'Generated/private path tracked in Git: {name}')
    verification=read(ROOT/'outputs/reproduced/verification.json')
    if verification.get('status')!='passed' or verification.get('output_records')!=8360:raise ValueError('Recorded-output reanalysis did not pass')
    smoke=read(ROOT/'outputs/smoke_training.json')
    if smoke.get('status')!='passed':raise ValueError('Historical tiny-model training smoke did not pass')
    prepared=read(ROOT/'work/evaluation_preparation.json')
    if prepared['case_ids_and_hashes_verified']!=1690:raise ValueError('Evaluation reconstruction did not pass')
    training_file=ROOT/'work/training_preparation.json'
    training=read(training_file) if training_file.exists() else {}
    # Current manuscript remains the supplied six-page venue build.
    try:
        import pymupdf
        pages=len(pymupdf.open(ROOT/'paper/main.pdf'))
        if pages!=6:raise ValueError('Manuscript is not six pages')
    except ImportError:pages='not checked; optional PyMuPDF unavailable'
    result={'status':'passed','files_verified':len(inventory),'credential_scan':'passed',
            'model_weights_or_restricted_input_datasets_committed':False,
            'evaluation_cases_reconstructed':1690,'recorded_outputs_rescored':8360,
            'training_split_bytes_reconstructed':training.get('exact_split_sha256_verified',False),
            'historical_tiny_model_training_smoke':'passed','full_7b_gpu_training_rerun':False,
            'fresh_7b_gpu_inference_rerun':False,'manuscript_pages':pages,
            'environment':{'python':platform.python_version(),**{n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','tokenizers']}}}
    write(ROOT/'reproducibility/release_verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
