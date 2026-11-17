"""Plot readable separation patterns and encoding interventions from frozen data."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE=Path(__file__).resolve().parent


def main():
    for name in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
        font_path=Path('C:/Windows/Fonts')/name
        if font_path.exists():font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({'font.family':('Times New Roman' if Path('C:/Windows/Fonts/times.ttf').exists() else 'DejaVu Serif'),'font.size':8,'axes.titlesize':8,'axes.labelsize':8,
                         'pdf.fonttype':42,'ps.fonttype':42})
    rows=list(csv.DictReader((HERE/'analysis/probe_scores.csv').open(encoding='utf-8')))
    results=json.loads((HERE/'analysis/results.json').read_text())
    methods=[('aside_orig','ASIDE','#536a7e'),('ha_aside','HA-ASIDE','#167c75')]
    fig,axes=plt.subplots(1,2,figsize=(7.14,2.65))
    fig.subplots_adjust(left=.06,right=.995,top=.88,bottom=.27,wspace=.28)
    for ax in axes:
        ax.set_axisbelow(True);ax.grid(axis='y',color='#e7e9eb',lw=.5)
        ax.spines[['top','right']].set_visible(False)
    x=np.arange(4);w=.34
    for offset,(method,label,color) in zip([-.5,.5],methods):
        paired={}
        for r in rows:
            if r['method']==method and r['role'] in {'user','data'}:
                paired.setdefault(r['source_row'],{})[r['role']]=r['form']=='True'
        assert len(paired)==200
        values=[100*np.mean([v['user'] and not v['data'] for v in paired.values()]),
                100*np.mean([v['user'] and v['data'] for v in paired.values()]),
                100*np.mean([not v['user'] and not v['data'] for v in paired.values()]),
                100*np.mean([not v['user'] and v['data'] for v in paired.values()])]
        bars=axes[0].bar(x+offset*w,values,width=w,color=color,label=label)
        axes[0].bar_label(bars,labels=[f'{v:.1f}' for v in values],padding=3,fontsize=8)
        iv=results['intervention_execution_rates']['form'][method]
        enc=[100*results['execution_rates']['form'][method]['data'],100*iv['user'],100*iv['system']]
        bars=axes[1].bar(np.arange(3)+offset*w,enc,width=w,color=color)
        axes[1].bar_label(bars,labels=[f'{v:.1f}' for v in enc],padding=3,fontsize=8)
    axes[0].set(ylim=(0,70),ylabel='Paired contexts (%)',title='(a) Outcomes for two separate responses')
    axes[0].set_xticks(x,['Selective\nanswering','Answers in\nboth inputs','Omits answer\nin both inputs','Reversed\nanswering'])
    axes[0].legend(loc='upper right',frameon=False,fontsize=8)
    axes[1].set(ylim=(0,100),ylabel='Answer-word mentions (%)',title='(b) Probe retained in the document')
    axes[1].set_xticks(np.arange(3),['Document\nencoding','Request\nencoding','Policy\nencoding'])
    for ext in ['pdf','png','svg']:
        fig.savefig(HERE/'figures'/('probe_diagnostics.'+ext),dpi=240)
    plt.close(fig)


if __name__=='__main__':main()
