"""Run the exact paper inputs with the shared archived inference implementation."""
import argparse,json,sys,platform,importlib.metadata
from pathlib import Path
from common import ROOT,read,write,lines,decode_prefix
from experiments.common import digest
from experiments.checkpoints import load_checkpoint,checkpoint_metadata
from experiments.rq_runner import generate_transcript

def selected_cases(cases,history_diagnostic=False):
    return [c for c in cases if c['suite']=='task_controls' or
            c['metadata']['condition'] in ({'system','user','data','assistant'} if history_diagnostic else {'system','user','data'})]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',required=True,choices=['vanilla','ise','aside_orig','ha_aside'])
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cases',type=Path,default=ROOT/'work/evaluation_cases.jsonl')
    p.add_argument('--source-root',type=Path,default=ROOT/'work/public_sources')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/inference')
    p.add_argument('--device',default='cuda:0');p.add_argument('--run',action='store_true')
    p.add_argument('--smoke',action='store_true',help='One authority and one utility case only')
    p.add_argument('--include-history-diagnostic',action='store_true',help='Archived auxiliary condition, outside the paper tables')
    args=p.parse_args();cases=selected_cases(list(lines(args.cases)),args.include_history_diagnostic)
    index=read(ROOT/'reproducibility/evaluation_cases.json')
    for c in cases:
        if digest(c)!=index[c['case_id']]['sha256']:raise ValueError(f"Input changed: {c['case_id']}")
    meta=checkpoint_metadata(args.checkpoint,args.method,'qwen25_7b')
    if args.method=='ha_aside':
        if meta['enable_rgag'] or meta['provenance'].get('pilot_max_steps') is not None:
            raise ValueError('Paper evaluation requires a complete RGAG-OFF training run')
        expected=read(ROOT/'reproducibility/training/reported_architecture.json')
        for key in ['base_revision','tokenizer_revision','training_data_sha256','validation_data_sha256']:
            if meta['provenance'][key]!=expected['provenance'][key]:raise ValueError(f'HA provenance differs: {key}')
        if meta['angles']!=expected['angles']:raise ValueError('HA role angles differ')
    if args.smoke:
        cases=[next(c for c in cases if c['suite']==suite) for suite in ['role_authority','task_controls']]
    plan=[]
    for c in cases:
        plan.append((c,None))
        if c['suite']=='role_authority' and c['metadata']['condition']=='data':
            plan.extend((c,encoding) for encoding in ['user','system'])
    print(json.dumps({'method':args.method,'generation_calls':len(plan),'smoke':args.smoke,'run':args.run}))
    if not args.run:return
    import torch
    from transformers import set_seed
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    args.output.mkdir(parents=True,exist_ok=True)
    output=args.output/(args.method+'.jsonl');manifest=args.output/(args.method+'.manifest.json')
    identity={'method':args.method,'checkpoint':str(args.checkpoint.resolve()),'checkpoint_metadata':meta,
              'case_hashes':{c['case_id']:digest(c) for c in cases},'plan':[(c['case_id'],e) for c,e in plan],
              'source_sha256':read(ROOT/'reproducibility/inference_source_sha256.json'),
              'device':args.device,'dtype':'bfloat16' if args.device.startswith('cuda') else 'float32',
              'seed':42,'greedy':True,'max_input_tokens':16384,'smoke':args.smoke}
    if manifest.exists() and read(manifest)['identity']!=identity:raise ValueError('Resume manifest differs')
    write(manifest,{'identity':identity,'environment':{'python':platform.python_version(),
          **{n:importlib.metadata.version(n) for n in ['torch','transformers','tokenizers','numpy']}}})
    cached={}
    if output.exists():
        for r in lines(output):
            key=(r['case_id'],r['encoding_intervention'])
            if key in cached:raise ValueError('Duplicate inference record')
            cached[key]=r
    allowed={(c['case_id'],e) for c,e in plan}
    if not set(cached)<=allowed:raise ValueError('Unexpected records in resume output')
    set_seed(42)
    model,tokenizer,_=load_checkpoint(args.checkpoint,args.method,'qwen25_7b',
             torch.bfloat16 if args.device.startswith('cuda') else torch.float32,args.device)
    config={'device':args.device,'seed':42,'max_input_tokens':16384,'_source_root':args.source_root}
    for c,e in plan:
        if (c['case_id'],e) in cached:continue
        result=generate_transcript(model,tokenizer,c['messages'],config,
                max_new_tokens=c.get('max_new_tokens',1024),tool=c.get('tool'),encoding_intervention=e)
        record={'method':args.method,'case_id':c['case_id'],'encoding_intervention':e,
                'case_sha256':digest(c),'output_ids':result['output_ids'],
                'prefix':decode_prefix(tokenizer,result['output_ids'])}
        with output.open('a',encoding='utf8',newline='\n') as f:
            f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
        cached[c['case_id'],e]=record
    print(json.dumps({'status':'complete','method':args.method,'generations':len(cached),'output':str(output)}))

if __name__=='__main__':main()
