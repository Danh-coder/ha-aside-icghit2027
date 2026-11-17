"""Regenerate the architecture, benchmark example, and result charts portably."""
import argparse,importlib.util
from pathlib import Path
from common import ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,default=ROOT/'outputs/reproduced')
    a=p.parse_args();(a.analysis/'figures').mkdir(parents=True,exist_ok=True)
    for name in ['make_rotation_figure','make_benchmark_figure','make_results_figure']:
        spec=importlib.util.spec_from_file_location(name,ROOT/'paper'/(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.HERE=a.analysis;module.main()
    print('Three figures regenerated in',a.analysis/'figures')

if __name__=='__main__':main()
