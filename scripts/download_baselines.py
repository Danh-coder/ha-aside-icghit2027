"""Preview or download the pinned published Vanilla, ASIDE and ISE-2 models."""
import argparse,json
from pathlib import Path
from common import ROOT
from experiments.published_checkpoints import PINS,prepare

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--download',action='store_true')
    p.add_argument('--method',action='append',choices=list(PINS));p.add_argument('--output',type=Path,default=ROOT/'checkpoints/baselines')
    a=p.parse_args()
    for m in a.method or list(PINS):print(json.dumps(prepare(m,a.output/m,download=a.download),indent=2))

if __name__=='__main__':main()
