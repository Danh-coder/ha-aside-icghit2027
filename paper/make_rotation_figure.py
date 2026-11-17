"""Draw the per-token role/embedding dataflow as a vector schematic."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle,Polygon,FancyArrowPatch,FancyBboxPatch,Circle

HERE=Path(__file__).resolve().parent
COLORS={'S':'#526471','U':'#237b9a','D':'#c67932'}

def main():
    font_dir=Path('C:/Windows/Fonts')
    for name in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
        if (font_dir/name).exists():font_manager.fontManager.addfont(str(font_dir/name))
    family='Times New Roman' if (font_dir/'times.ttf').exists() else 'DejaVu Serif'
    plt.rcParams.update({'font.family':family,'font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
    fig=plt.figure(figsize=(7.14,2.05));ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def text(x,y,s,**kw):return ax.text(x,y,s,fontsize=8,**kw)
    def arrow(a,b,color='#4a5965',dashed=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=8,
                     linewidth=.8,color=color,linestyle='--' if dashed else '-'))
    def document(x,y,color,w=.036,h=.105):
        fold=.011
        ax.add_patch(Polygon([(x,y),(x+w,y),(x+w,y+h-fold),(x+w-fold,y+h),(x,y+h)],
                            closed=True,facecolor='white',edgecolor=color,lw=.9))
        ax.plot([x+w-fold,x+w-fold,x+w],[y+h,y+h-fold,y+h-fold],lw=.6,color=color)
        for yy in [y+.075,y+.052,y+.029]:ax.plot([x+.006,x+w-.006],[yy,yy],lw=.6,color=color)
    for x,label in [(.09,'Input sources'),(.272,'Token sequence'),(.424,'Shared lookup E'),
                    (.62,'Pairwise rotation'),(.864,'Shared Transformer')]:
        text(x,.955,label,ha='center',va='top')
    ax.add_patch(FancyBboxPatch((.211,.20),.119,.64,boxstyle='round,pad=0.006',
                 facecolor='none',edgecolor='#aab5bd',lw=.6))
    for center,role,label in [(.755,'S','Application policy'),(.515,'U','User request'),(.275,'D','External document')]:
        c=COLORS[role];document(.012,center-.046,c)
        text(.056,center+.025,label,ha='left',va='center')
        text(.056,center-.04,{'S':'SYSTEM','U':'USER','D':'DATA'}[role],color=c,ha='left',va='center')
        arrow((.173,center),(.216,center),c)
        ax.add_patch(Rectangle((.218,center-.05),.055,.10,facecolor=c,alpha=.13,edgecolor=c,lw=.6))
        ax.add_patch(Rectangle((.283,center-.05),.040,.10,facecolor='white',edgecolor=c,lw=.6))
        text(.2455,center,'token',ha='center',va='center')
        text(.303,center,role,color=c,ha='center',va='center')
    text(.2455,.15,'tokens',ha='center');text(.303,.15,'roles',ha='center')
    ax.plot([.33,.345,.345,.51],[.84,.84,.865,.865],color='#526471',lw=.7,ls='--')
    arrow((.51,.865),(.61,.865),dashed=True)
    text(.43,.79,'source label rᵢ',ha='center',va='center')
    text(.62,.79,'select angle θ',ha='center',va='center')
    arrow((.33,.515),(.382,.515))
    for row in range(5):
        for col in range(4):
            ax.add_patch(Rectangle((.387+col*.018,.36+row*.057),.017,.052,
                         facecolor='#c5d8e0' if row==2 else '#edf0f2',edgecolor='#8d9aa3',lw=.45))
    text(.424,.27,'embedding eᵢ',ha='center')
    arrow((.465,.515),(.505,.515))
    ax.add_patch(FancyBboxPatch((.51,.22),.222,.52,boxstyle='round,pad=0.008',
                 facecolor='#f6f8f9',edgecolor='#8d9aa3',lw=.65))
    for y,role,label in [(.65,'S','S: 0'),(.48,'U','U: π/8'),(.31,'D','D: π/2')]:
        c=COLORS[role]
        for x in [.522,.544,.680,.702]:
            ax.add_patch(Rectangle((x,y-.033),.017,.066,facecolor=c,alpha=.28,edgecolor=c,lw=.6))
        arrow((.567,y),(.672,y),c)
        text(.62,y+.042,label,color=c,ha='center',va='bottom')
    text(.62,.145,'Same angle for all coordinate pairs',ha='center')
    text(.62,.075,'Clockwise; no added trainable parameters',ha='center')
    arrow((.74,.515),(.786,.515))
    ax.add_patch(FancyBboxPatch((.793,.275),.150,.56,boxstyle='round,pad=0.007',
                 facecolor='white',edgecolor='#526471',lw=.8))
    for k in [2,1,0]:
        ax.add_patch(FancyBboxPatch((.806+.009*k,.48+.05*k),.111,.145,
                     boxstyle='round,pad=0.004',facecolor='#e5edf1',edgecolor='#526471',lw=.7))
    text(.861,.55,'Decoder layers',ha='center',va='center')
    text(.868,.32,'Fine-tuned',ha='center',va='center')
    arrow((.865,.26),(.865,.135))
    document(.847,.016,'#526471',w=.036,h=.105)
    text(.895,.067,'Response',ha='left',va='center')
    for ext in ['pdf','png','svg']:fig.savefig(HERE/'figures'/('overall_architecture.'+ext),dpi=260)
    plt.close(fig)

if __name__=='__main__':main()
