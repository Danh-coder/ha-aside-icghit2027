"""Launch the exact historical first training configuration (not a search)."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from common import ROOT,read,sha

def command(output,smoke=False):
    config=read(ROOT/'configs/training.json')
    code=ROOT/'reproducibility/historical_training/ha-aside-paper'
    reference=read(ROOT/'reproducibility/training/reported_architecture.json')
    for name,expected in reference['provenance']['source_sha256'].items():
        if sha(code/name)!=expected:raise ValueError(f'Historical training code changed: {name}')
    script=code/'training/train_ha_aside.py'
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1','--',str(script),
         '--model','qwen25_7b','--method','ha_aside','--dataset','chatqa_multirole','--no-rgag',
         '--single','--base-revision',config['base_revision'],'--tokenizer-revision',config['tokenizer_revision'],
         '--seed','42','--micro-batch-size','1','--gradient-checkpointing',
         '--data-dir',str(ROOT/'ha-aside-paper/data/finetuning/chatqa_multirole_v1'),
         '--deepspeed',str(ROOT/'configs/deepspeed_zero2_cpu_offload.json'),'--output',str(output)]
    if smoke:cmd+=['--pilot-steps','2','--pilot-effective-batch-size','2']
    return cmd

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true')
    p.add_argument('--verify-data',action='store_true',help='Execute the historical trainer dry run; load no model weights')
    p.add_argument('--smoke',action='store_true',help='Two optimizer steps; never paper training')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/training')
    args=p.parse_args();cmd=command(args.output,args.smoke)
    print(json.dumps({'command':cmd+(['--run'] if args.run else []),
          'paper_recipe':not args.smoke,'runs_hyperparameter_search':False},indent=2))
    if args.run:subprocess.run(cmd+['--run'],cwd=ROOT,check=True)
    elif args.verify_data:
        # Distributed launch is unnecessary for the data/configuration dry run.
        start=cmd.index('--model')
        subprocess.run([sys.executable,cmd[start-1],*cmd[start:]],cwd=ROOT,check=True)

if __name__=='__main__':main()
