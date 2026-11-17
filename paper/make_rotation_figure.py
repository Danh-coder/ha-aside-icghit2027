"""Illustrate source pages, coordinate pairs, rotation geometry, and model flow."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch, Circle, Arc

HERE=Path(__file__).resolve().parent
COLORS={'policy':'#526471','request':'#237b9a','document':'#c67932'}


def main():
    for name in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
        font_path=Path('C:/Windows/Fonts')/name
        if font_path.exists():font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({'font.family':('Times New Roman' if Path('C:/Windows/Fonts/times.ttf').exists() else 'DejaVu Serif'),'font.size':8,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig=plt.figure(figsize=(7.14,2.85))
    ax=fig.add_axes([.01,.01,.98,.98])
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')

    def arrow(p,q,color='#35434c',width=.8):
        ax.add_patch(FancyArrowPatch(p,q,arrowstyle='->',mutation_scale=9,
                                    linewidth=width,color=color))

    for y,key,label in [(.74,'policy','Application policy'),
                         (.49,'request','User request'),
                         (.24,'document','External document')]:
        color=COLORS[key]
        x=.025;w=.12;h=.15;fold=.022
        ax.add_patch(Polygon([(x,y),(x+w,y),(x+w,y+h-fold),
                              (x+w-fold,y+h),(x,y+h)],closed=True,
                             facecolor='white',edgecolor=color,linewidth=1))
        ax.plot([x+w-fold,x+w-fold,x+w],[y+h,y+h-fold,y+h-fold],color=color,lw=.7)
        for ly,length in [(y+.105,.066),(y+.077,.080),(y+.049,.056)]:
            ax.plot([x+.017,x+.017+length],[ly,ly],color=color,lw=.75)
        ax.text(x+w/2,y-.055,label,ha='center',va='top',fontsize=8)
        arrow((x+w+.008,y+.075),(.225,y+.075),color)
        # Two rows depict successive coordinate pairs in a token embedding.
        for k in range(4):
            for j in range(2):
                ax.add_patch(Rectangle((.23+k*.027,y+.018+j*.048),.022,.039,
                             edgecolor=color,facecolor=color,alpha=.20+.14*j,lw=.6))
        for k in range(4):
            xx=.23+k*.027
            ax.plot([xx,xx,xx+.022,xx+.022],[y+.122,y+.133,y+.133,y+.122],
                    color=color,lw=.6)
    ax.text(.282,.955,'Token embedding',ha='center',fontsize=8)
    ax.text(.282,.115,'Adjacent coordinate pairs',ha='center',fontsize=8)
    arrow((.355,.56),(.407,.56))

    # A separate equal-aspect axis ensures that the visual angles are accurate.
    geo=fig.add_axes([.414,.24,.235,.61])
    geo.set(xlim=(-.25,1.35),ylim=(-1.22,.48),aspect='equal');geo.axis('off')
    geo.annotate('',xy=(1.27,0),xytext=(-.16,0),
                 arrowprops={'arrowstyle':'->','linewidth':.6,'color':'#a4a9ad'})
    geo.annotate('',xy=(0,.40),xytext=(0,-1.14),
                 arrowprops={'arrowstyle':'->','linewidth':.6,'color':'#a4a9ad'})
    geo.add_patch(Arc((0,0),1.28,1.28,theta1=-90,theta2=0,
                      color='#b3b8bb',lw=.7,linestyle=':'))
    for theta,key in [(0,'policy'),(np.pi/8,'request'),(np.pi/2,'document')]:
        end=(np.cos(theta),-np.sin(theta))
        geo.annotate('',xy=end,xytext=(0,0),
                     arrowprops={'arrowstyle':'->','linewidth':1.6,'color':COLORS[key]})
    geo.text(.87,.14,'Policy: 0',color=COLORS['policy'],ha='center',fontsize=8)
    geo.text(.72,-.62,'Request: π/8',color=COLORS['request'],ha='center',fontsize=8)
    geo.text(.18,-1.15,'Document: π/2',color=COLORS['document'],ha='left',fontsize=8)
    ax.text(.535,.955,'Same vector, role-selected angle',ha='center',fontsize=8)
    ax.text(.535,.115,'Repeat for every coordinate pair',ha='center',fontsize=8)
    arrow((.655,.56),(.70,.56))

    # Shared neural layers: all role-conditioned vectors use one model.
    for x in [.735,.79,.845]:
        for y in [.34,.49,.64,.79]:
            ax.add_patch(Circle((x,y),.014,edgecolor='#526471',facecolor='#edf1f3',lw=.8))
    for x1,x2 in [(.735,.79),(.79,.845)]:
        for y1 in [.34,.49,.64,.79]:
            for y2 in [.34,.49,.64,.79]:
                ax.plot([x1+.014,x2-.014],[y1,y2],color='#c0c8cd',lw=.35,zorder=-1)
    ax.text(.79,.955,'Shared Transformer',ha='center',fontsize=8)
    ax.text(.79,.115,'Existing model parameters',ha='center',fontsize=8)
    arrow((.873,.56),(.92,.56))
    ax.add_patch(Polygon([(.928,.36),(.989,.36),(.989,.73),(.975,.76),(.928,.76)],
                         closed=True,edgecolor='#526471',facecolor='white',lw=.8))
    for y in [.67,.61,.55,.49,.43]:
        ax.plot([.938,.98],[y,y],color='#526471',lw=.6)
    ax.text(.958,.24,'Response',ha='center',fontsize=8)
    for ext in ['pdf','png','svg']:
        fig.savefig(HERE/'figures'/('overall_architecture.'+ext),dpi=240)
    plt.close(fig)


if __name__=='__main__':main()
