"""Reconstruct the exact train/validation bytes from pinned ChatQA source rows."""
import argparse,json,shutil,hashlib
from pathlib import Path
from common import ROOT,read,lines,download,sha,write
from experiments.common import digest
from tools.prepare_chatqa_dataset import load_prepared

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,help='Optional directory with synthetic_convqa_train.json and README.md')
    p.add_argument('--output',type=Path,default=ROOT/'ha-aside-paper/data/finetuning/chatqa_multirole_v1')
    args=p.parse_args();ref=ROOT/'reproducibility/training';manifest=read(ref/'manifest.json')
    cache=ROOT/'work/chatqa_source';revision=manifest['source_revision']
    prefix=f'https://huggingface.co/datasets/nvidia/ChatQA-Training-Data/resolve/{revision}'
    for name,remote,expected in [('synthetic_convqa_train.json','synthetic_convqa/train.json',manifest['source_sha256']),
        ('README.md','README.md',manifest['files']['SOURCE_CARD.md']['sha256'])]:
        download(f'{prefix}/{remote}',cache/name,expected,args.cache_dir/name if args.cache_dir else None)
    raw=read(cache/'synthetic_convqa_train.json');indices=list(lines(ref/'source_rows.jsonl.gz'))
    if len(indices)!=35598:raise ValueError('Unexpected retained-row count')
    args.output.mkdir(parents=True,exist_ok=True)
    for split in ['train','validation']:
        target=args.output/(split+'.jsonl');part=target.with_suffix('.jsonl.partial')
        with part.open('w',encoding='utf8',newline='\n') as f:
            for entry in indices:
                if entry['split']!=split:continue
                source=raw[entry['source_row']]
                if digest(source)!=entry['source_row_sha256']:raise ValueError('Source row hash mismatch')
                metadata={k:v for k,v in entry.items() if k!='split'}
                f.write(json.dumps({**metadata,**source},ensure_ascii=False)+'\n')
        if sha(part)!=manifest['files'][target.name]['sha256']:raise ValueError(f'Exact split hash mismatch: {split}')
        if target.exists() and sha(target)!=sha(part):raise FileExistsError(target)
        part.replace(target)
    for name in ['manifest.json','filter_report.json','semantic_screening.json']:
        shutil.copyfile(ref/name,args.output/name)
    shutil.copyfile(cache/'README.md',args.output/'SOURCE_CARD.md')
    checked,_=load_prepared(args.output)
    result={'status':'passed','source_revision':revision,'train_rows':32046,'validation_rows':3552,
            'exact_split_sha256_verified':True,'output':str(args.output)}
    write(ROOT/'work/training_preparation.json',result);print(json.dumps(result))

if __name__=='__main__':main()
