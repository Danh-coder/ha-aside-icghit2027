"""Download pinned sources and reconstruct only the paper's frozen case IDs."""
import argparse,concurrent.futures,json,hashlib
from pathlib import Path
from common import ROOT,read,write,download,longpath
from experiments.common import digest
from experiments.rq_data import iheval_cases,mtrag_tasks,authority_cases,task_control_cases

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,help='Optional verified copy of the upstream source tree')
    p.add_argument('--source-root',type=Path,default=ROOT/'work/public_sources')
    p.add_argument('--output',type=Path,default=ROOT/'work/evaluation_cases.jsonl')
    args=p.parse_args();args.source_root=longpath(args.source_root)
    if args.cache_dir:args.cache_dir=longpath(args.cache_dir)
    lock=read(ROOT/'reproducibility/sources.lock.json')
    selected={k:v for k,v in lock['files'].items() if k.startswith(('iheval/','mtrag/','bipia/')) or k=='local/sep.json'}
    def fetch(item):
        name,spec=item
        if name=='local/sep.json':
            url='https://raw.githubusercontent.com/egozverev/aside/dfa94270d48b006decdcd4d455df3e10d6004880/experiments/data/SEP_dataset.json'
        else:
            repo,relative=name.split('/',1);pin=lock['repositories'][repo]
            url=f"https://raw.githubusercontent.com/{pin['repository']}/{pin['revision']}/{relative}"
        download(url,args.source_root/name,spec['sha256'],args.cache_dir/name if args.cache_dir else None)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(fetch,selected.items()))
    write(args.source_root/'sources.lock.json',lock)
    index=read(ROOT/'reproducibility/evaluation_cases.json')
    native=list(iheval_cases(args.source_root,lock));tasks=mtrag_tasks(args.source_root,lock)
    found={}
    for case in [*authority_cases(args.source_root,lock,tasks,native),*task_control_cases(args.source_root,lock,native,tasks)]:
        if case['case_id'] not in index:continue
        if digest(case)!=index[case['case_id']]['sha256']:raise ValueError(f"Case changed: {case['case_id']}")
        found[case['case_id']]=case
    if set(found)!=set(index):raise ValueError('Missing or unexpected selected benchmark cases')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    encoded=''.join(json.dumps(found[cid],ensure_ascii=False)+'\n' for cid in sorted(found))
    if args.output.exists() and args.output.read_text(encoding='utf8')!=encoded:raise FileExistsError(args.output)
    args.output.write_text(encoded,encoding='utf8',newline='\n')
    write(ROOT/'work/evaluation_preparation.json',{'status':'passed','authority_cases':800,'utility_cases':890,
           'contexts':200,'case_ids_and_hashes_verified':len(found),'source_files_verified':len(selected)})
    print(json.dumps({'status':'passed','verified_cases':len(found),'output':str(args.output)}))

if __name__=='__main__':main()
