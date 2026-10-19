"""Publication figures from pinned, task-local result tables and saved predictions."""
from pathlib import Path
import hashlib,json,textwrap
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.transforms import Bbox
from matplotlib.text import Text
from io import StringIO
from svg_schematics import write_sources, integrate, render
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle,Patch
from scipy.special import softmax
from sklearn.metrics import roc_curve
from figure_layouts import SIZES, RECTS, TITLES, LABELS

ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence';F=ROOT/'figures';F.mkdir(exist_ok=True)
P=pd.read_csv(E/'deep_type_level.csv');D=pd.read_csv(E/'historical-all-donor-pairs.csv');DS=D[D.status=='scored'].copy()
S=pd.read_csv(E/'historical-specifications.csv');FM=pd.read_csv(E/'historical-frozen-representations.csv');EX=pd.read_csv(E/'historical-portability.csv')
N=pd.read_csv(E/'new-score-comparisons.csv');O=pd.read_csv(E/'new-locked-thresholds.csv');C=pd.read_csv(E/'new-known-calibration.csv')
NC=pd.read_csv(E/'new-label-controls.csv');PV=pd.read_csv(E/'new-prevalence-sensitivity.csv');ST=pd.read_csv(E/'rank-shift-stress.csv')
AG=json.loads((E/'aggregation-extension.json').read_text());HS=json.loads((E/'historical-summary.json').read_text())
OM=pd.read_csv(E/'omission-sensitivity.csv');W=pd.read_csv(E/'atlas-weight-sensitivity.csv')
RE=json.loads((E/'revision_evidence.json').read_text());HC=json.loads((E/'historical-cross-atlas-controls.json').read_text())
ORDER=['Pulmonary','Hepatic','Adult cardiac'];SHORT=['Pulmonary','Hepatic','Adult cardiac']
COL={'Pulmonary':'#218572','Hepatic':'#BA526C','Adult cardiac':'#396FAC'}
METHODS=['msp_raw','msp_temperature','entropy_raw','entropy_temperature','centroid_distance']
MSHORT=['MSP','MSP\n+ T','Ent.','Ent.\n+ T','Dist.']
MN={'msp_raw':'MSP','msp_temperature':'MSP + T','entropy_raw':'Entropy','entropy_temperature':'Entropy + T','centroid_distance':'Centroid'}
MC=dict(zip(METHODS,['#575D68','#218572','#C28B29','#6F55A5','#B94E5D']))
MFO=['Geneformer','Geneformer-316','scGPT','scFoundation','UCE','CellPLM'];MFL=['GF','GF large','scGPT','scFound.','UCE','CellPLM']
SEED=20261016
plt.rcParams.update({'font.family':'Arial','font.size':10.5,'axes.labelsize':11.2,'axes.titlesize':11.8,
 'xtick.labelsize':10.4,'ytick.labelsize':10.4,'legend.fontsize':9.8,'pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','svg.hashsalt':'array-05851-enhanced','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
 'xtick.major.width':.6,'ytick.major.width':.6,'savefig.facecolor':'white','figure.facecolor':'white'})
MANIFEST=[]
# These are panel-specific header allocations in points, not a common grid.
# Legends occupy the original panel's upper strip; the data frame ends below it.
LEGEND_HEADERS={
 '1':{'B':(3,20),'D':(2,20),'F':(1,32)},
 '2':{'A':(3,20)},
 '3':{'B':(2,32),'D':(2,20),'E':(2,20)},
 '4':{'G':(2,32)},
 '5':{'B':(2,44),'F':(2,20)},
 '6':{'B':(3,32),'D':(2,32)},
 '7':{'A':(3,20),'B':(3,34),'H':(2,32)},
 '8':{'C':(2,44)},
 '9':{'E':(3,20),'F':(2,32)}}
NOTE_HEADERS={'1':{'G':20},'3':{'A':20,'C':20},
              '4':{'E':20},'5':{'H':32},'7':{'F':32},'9':{'G':20}}
LAYOUT_CHECKS=[]

def panel_headers(fig,number):
    """Place legends/statistics outside data frames in measured, reserved strips."""
    axes=fig._main_axes
    for i,ax in enumerate(axes):
        letter=chr(65+i);old=ax.get_position().frozen()
        spec=LEGEND_HEADERS.get(number,{}).get(letter)
        reserve=spec[1] if spec else NOTE_HEADERS.get(number,{}).get(letter,0)
        # D--F share three covariate rows and therefore retain aligned frames.
        if number=='3' and letter=='F':reserve=20
        if not reserve:continue
        delta=reserve/(fig.get_figheight()*72)
        ax.set_position([old.x0,old.y0,old.width,old.height-delta])
        height=ax.get_position().height
        ax.set_title(ax.get_title(),y=1+delta/height,pad=7,
                     loc='center',fontweight='normal',fontsize=11.8)
        if spec:
            legend=ax.get_legend();assert legend is not None,(number,letter)
            labels=[t.get_text() for t in legend.get_texts()]
            handles,source_labels=ax.get_legend_handles_labels()
            if not handles or source_labels!=labels:handles=legend.legend_handles
            if number in ['3','4']:
                labels=[s.replace('Adult cardiac','Cardiac') for s in labels]
            if number=='9' and letter=='F':
                labels=[s.removesuffix(' context') for s in labels]
            title=legend.get_title().get_text()
            legend.remove()
            legend=ax.legend(handles,labels,loc='lower center',
                bbox_to_anchor=(.5,1+3/(height*fig.get_figheight()*72)),
                ncol=spec[0],frameon=False,fontsize=9.8,
                title=title or None,title_fontsize=9.8,
                handlelength=1.15,handletextpad=.35,columnspacing=.8,
                labelspacing=.25,borderpad=0,borderaxespad=0)
            ax._header_legend=legend
        else:
            for text in ax.texts:
                if number in ['4','9']:
                    x=text.get_position()[0]
                    text.set_transform(ax.get_xaxis_transform())
                    text.set_position((x,1+4/(height*fig.get_figheight()*72)))
                    text.set_ha('center');text.set_va('bottom')
                else:
                    text.set_transform(ax.transAxes)
                    text.set_position((.5,1+4/(height*fig.get_figheight()*72)))
                    text.set_ha('center');text.set_va('bottom')
                text.set_bbox(None)
                text.set_clip_on(False)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    frames=[a.get_window_extent(renderer) for a in axes]
    records=[];notes=[]
    for i,ax in enumerate(axes):
        legend=ax.get_legend()
        if legend is None:continue
        bbox=legend.get_window_extent(renderer)
        overlap=[chr(65+j) for j,frame in enumerate(frames)
                 if bbox.overlaps(frame)]
        assert not overlap,(number,chr(65+i),'legend enters data frame',overlap)
        width=frames[i].width
        assert bbox.width<=width+1,(number,chr(65+i),'legend wider than panel',bbox.width,width)
        records.append({'panel':chr(65+i),'legend_labels':[t.get_text() for t in legend.get_texts()],
                        'legend_bbox_pixels':list(map(float,bbox.extents)),
                        'data_frame_intersections':overlap})
    for i,ax in enumerate(axes):
        if chr(65+i) not in NOTE_HEADERS.get(number,{}):continue
        for text in ax.texts:
            bbox=text.get_window_extent(renderer)
            overlap=[chr(65+j) for j,frame in enumerate(frames) if bbox.overlaps(frame)]
            assert not overlap,(number,chr(65+i),'note enters data frame',text.get_text(),overlap)
            notes.append({'panel':chr(65+i),'text':text.get_text(),
                          'data_frame_intersections':overlap})
    LAYOUT_CHECKS.append({'figure':number,'legends':records,
                          'statistical_notes':notes,
                          'legends_clear_of_every_data_frame':True})

def plate(number,titles,sources):
    fig=plt.figure(figsize=SIZES[number]);axes=[]
    for k,(title,rect) in enumerate(zip(TITLES[number],RECTS[number])):
        x,y,w,h=rect;ax=fig.add_axes(rect);axes.append(ax)
        ax._panel_letter=fig.text(x-.038,y+h+.013,chr(65+k),fontsize=15,fontweight='bold',ha='left',va='bottom')
        wrap=40 if w>=.38 else 25 if w>=.27 else 21
        ax.set_title(textwrap.fill(title,wrap),pad=7,loc='center',fontweight='normal',fontsize=11.8)
        ax.tick_params(length=3,pad=2.5 if str(number).startswith('S') else 4.5)
        ax.grid(axis='y',color='#E9ECEF',lw=.5,zorder=0);ax.set_axisbelow(True)
    fig._main_axes=axes
    MANIFEST.append({'figure':number,'layout':'hand-positioned by panel content','figure_inches':list(SIZES[number]),
                     'panels':[{'panel':chr(65+i),'title':TITLES[number][i],'sources':sources,'axes_rect':list(RECTS[number][i])} for i in range(len(titles))]})
    return fig,axes

def save(fig,number):
    # Geometry is explicit above. Wrap only descriptive labels that would exceed
    # their allocated frame; do not alter values, limits, ticks or scientific data.
    for ax in getattr(fig,'_main_axes',fig.axes):
        width=ax.get_position().width
        for getter,setter in [(ax.get_xlabel,ax.set_xlabel),(ax.get_ylabel,ax.set_ylabel)]:
            label=LABELS.get(getter(),getter())
            if getter==ax.get_xlabel and width<.28:
                wrap=20 if width<.22 else 25
                if len(label)>wrap:label=textwrap.fill(label,wrap)
            setter(label)
    # A minimum 9.8 pt on the 8.4-inch plate becomes 8.2 pt at the
    # manuscript's 7.06-inch width. Only panel letters use bold weight.
    letters={getattr(a,'_panel_letter',None) for a in getattr(fig,'_main_axes',fig.axes)}
    for artist in fig.findobj(Text):
        if artist not in letters:
            artist.set_fontweight('normal')
            if artist.get_fontsize()<9.8:artist.set_fontsize(9.8)
    if not str(number).startswith('S'):panel_headers(fig,number)
    for ext in ['pdf','svg','png']:
        kw={'dpi':300} if ext=='png' else {}
        if ext=='pdf':kw['metadata']={'CreationDate':None,'ModDate':None}
        if ext=='svg':kw['metadata']={'Date':None}
        fig.savefig(F/f'Figure{number}.{ext}',**kw)
    if number=='1':
        write_sources(ROOT)
        path=F/'Figure1.svg';path.write_text(integrate(path.read_text(),ROOT,RECTS['1'],SIZES['1']))
        render(path,F/'Figure1.pdf',F/'Figure1.png')
    # Separate vector panels support editorial production without flattening
    # the composite plate. Cropping never changes the plotted data or domains.
    out=F/'panels';out.mkdir(exist_ok=True)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();panel_records=[]
    main_axes=getattr(fig,'_main_axes',list(fig.axes))
    for i,ax in enumerate(main_axes):
        associated=[ax]+([ax._colorbar_axes] if hasattr(ax,'_colorbar_axes') else [])
        letter=ax._panel_letter
        boxes=[a.get_tightbbox(renderer) for a in associated]+[letter.get_window_extent(renderer)]
        bbox=Bbox.union(boxes).expanded(1.018,1.018).transformed(fig.dpi_scale_trans.inverted())
        outside=bool(bbox.x0<-.001 or bbox.y0<-.001 or bbox.x1>fig.get_figwidth()+.001 or bbox.y1>fig.get_figheight()+.001)
        states=[(a,a.get_visible()) for a in fig.axes]+[(t,t.get_visible()) for t in fig.texts]
        for a in fig.axes:a.set_visible(a in associated)
        for t in fig.texts:t.set_visible(t is letter)
        path=out/f'Figure{number}{chr(65+i)}.pdf'
        if number=='1' and chr(65+i) in ['A','C','E']:
            temp=F/'schematics'/f'panel-{chr(65+i)}-composite.svg'
            buffer=StringIO();fig.savefig(buffer,format='svg',metadata={'Date':None})
            temp.write_text(integrate(buffer.getvalue(),ROOT,RECTS['1'],SIZES['1'],only=chr(65+i)))
            crop=[bbox.x0*72,(fig.get_figheight()-bbox.y1)*72,bbox.width*72,bbox.height*72]
            render(temp,path,crop=crop);temp.unlink()
        else:fig.savefig(path,bbox_inches=bbox,pad_inches=0,metadata={'CreationDate':None,'ModDate':None})
        for artist,visible in states:artist.set_visible(visible)
        panel_records.append({'panel':chr(65+i),'vector_file':str(path.relative_to(ROOT)),
                              'bbox_inches':list(map(float,bbox.extents)),'outside_plate':outside,
                              'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    MANIFEST[-1]['panel_exports']=panel_records
    MANIFEST[-1]['typography']={'panel_title_weight':'normal','panel_title_alignment':'center','panel_letter_weight':'bold','minimum_matplotlib_text_pt':9.8,'main_page_width_in':7.06,'minimum_scaled_matplotlib_text_pt':9.8*7.06/8.4}
    MANIFEST[-1]['data_frame_fraction']=sum(a.get_position().width*a.get_position().height for a in main_axes)
    plt.close(fig)

def diagonal(ax,lo=0,hi=1):
    ax.plot([lo,hi],[lo,hi],color='#89929E',lw=.8,ls='--',zorder=1);ax.set_xlim(lo,hi);ax.set_ylim(lo,hi)

def hchance(ax):ax.axhline(.5,color='#6C727B',lw=.75,ls='--')
def vchance(ax):ax.axvline(.5,color='#6C727B',lw=.75,ls='--')

def heat(ax,frame,vmin=0,vmax=1,cmap='viridis',fmt=None,label='AUROC',labelsize=7,
         compact_numbers=None,adaptive_contrast=False,colorbar_title_pad=10,
         horizontal_colorbar=False,scale_heading_above_frame=False):
    v=frame.to_numpy(float);cm=plt.get_cmap(cmap).copy();cm.set_bad('#F0F2F4')
    im=ax.imshow(v,vmin=vmin,vmax=vmax,cmap=cm,aspect='auto',interpolation='nearest')
    ax.set_xticks(range(frame.shape[1]),frame.columns);ax.set_yticks(range(frame.shape[0]),frame.index)
    ax.tick_params(axis='both',labelsize=labelsize,length=0,pad=3);ax.grid(False)
    ax.set_xticks(np.arange(-.5,frame.shape[1],1),minor=True);ax.set_yticks(np.arange(-.5,frame.shape[0],1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.7);ax.tick_params(which='minor',length=0)
    if fmt:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if np.isfinite(v[i,j]):
                    fraction=(v[i,j]-vmin)/(vmax-vmin)
                    compact=frame.shape[1]>=5 if compact_numbers is None else compact_numbers
                    text=format(v[i,j],fmt)
                    if compact:text=text.replace('-0.','−.').replace('0.','.',1)
                    foreground='white' if fraction<.45 else '#1A232B'
                    if adaptive_contrast:
                        # Both ends of RdBu are dark. Choose black/white from
                        # the displayed cell color rather than its signed value.
                        rgb=np.array(cm(im.norm(v[i,j]))[:3])
                        linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
                        luminance=linear @ np.array([.2126,.7152,.0722])
                        foreground='white' if 1.05/(luminance+.05)>(luminance+.05)/.05 else 'black'
                    ax.text(j,i,text,ha='center',va='center',fontsize=6.8,color=foreground)
                else:ax.text(j,i,'-',ha='center',va='center',color='#818A94')
    pos=ax.get_position()
    cax=ax.figure.add_axes([pos.x0,.055,pos.width,.009] if horizontal_colorbar
                          else [pos.x1+.008,pos.y0+.025,.008,pos.height-.05])
    ax._colorbar_axes=cax
    cb=ax.figure.colorbar(im,cax=cax,orientation='horizontal' if horizontal_colorbar else 'vertical')
    cb.outline.set_visible(False);cb.ax.tick_params(labelsize=6.5,length=2,pad=2)
    # Keep scale headings in the reserved colorbar margin, never over data.
    heading='log10\n(n+1)' if label=='log10(n+1)' else label
    if horizontal_colorbar:
        cb.set_ticks([-1,0,1]);cb.ax.set_xlabel(heading,fontsize=9.8,labelpad=1)
    elif scale_heading_above_frame:
        cb.ax.set_title(heading,fontsize=9.8,pad=4,loc='center',
                       y=1+.025/cax.get_position().height)
    else:cb.ax.set_title(heading,fontsize=6.5,pad=colorbar_title_pad,loc='left')
    return im

def distribution(ax,frame,column,group,groups,colorset=None,labels=None,limits=None):
    rng=np.random.default_rng(SEED)
    for j,g in enumerate(groups):
        x=frame.loc[frame[group]==g,column].dropna().to_numpy();color=colorset[g] if colorset else '#396FAC'
        if not len(x):continue
        ax.scatter(j+rng.uniform(-.19,.19,len(x)),x,s=10,color=color,alpha=.42,linewidth=0,zorder=2)
        q=np.quantile(x,[.25,.5,.75]);ax.plot([j,j],[q[0],q[2]],lw=4,color=color,zorder=3)
        ax.plot([j-.13,j+.13],[q[1],q[1]],lw=1.6,color='black',zorder=4)
    ax.set_xticks(range(len(groups)),labels if labels else groups);ax.set_xlim(-.5,len(groups)-.5)
    if limits:ax.set_ylim(*limits)

def curves(ax,frame,x,y,group,groups,colormap,labels=None,band=True):
    for g in groups:
        d=frame[frame[group]==g];summary=d.groupby(x)[y].quantile([.25,.5,.75]).unstack()
        xx=summary.index.to_numpy(float);ax.plot(xx,summary[.5],'-o',ms=3,lw=1.25,color=colormap[g],label=labels[g] if labels else g)
        if band:ax.fill_between(xx,summary[.25],summary[.75],alpha=.09,color=colormap[g],lw=0)

def box(ax,x,y,w,h,text,color='#E7EDF2',fontsize=8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=.02',fc=color,ec='#8B969F',lw=.7,clip_on=False))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize)

def arrow(ax,start,end):ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=9,color='#52616E',lw=.9))

def anchor_row(target='Ciliated'):
    d=N[(N.type==target)&(N.test_donor=='ASK440')&(N.calibration_donor=='ASK452')&(N.method=='msp_temperature')].iloc[0]
    p=ROOT/'experiments/SC31/rows'/f'{d.row_id}.json';j=json.loads(p.read_text())
    z=np.load(p.parent.parent/'scores'/f'{d.row_id}.npz');return j,z

def figure1():
    titles=['Donor roles and target exclusion','Primary atlas inventory','Where fitting is permitted','Support is checked before scoring',
            'Matched novelty scores','Confidence in an example target','Temperature chosen on calibration','Donor composition in the extension']
    fig,ax=plate('1',titles,['deep_type_level.csv','SC31/input_inventory.json','SC31/scores'])
    a=ax[0];a.axis('off');a.set_xlim(0,1);a.set_ylim(0,1)
    a=ax[1];data=np.array([[4561,2485,3314],[3612,3127,1705],[19753,4981,5045]])
    left=np.zeros(3)
    for k,(label,col) in enumerate(zip(['Train','Calibration','Held'],['#647F8B','#D6B46A','#558BC1'])):
        a.barh(range(3),data[:,k],left=left,color=col,label=label,height=.6);left+=data[:,k]
    a.set_yticks(range(3),ORDER);a.set_xlabel('Cells before target exclusion');a.legend(loc='lower right',frameon=False,ncol=1)
    for i,n in enumerate(left):a.text(n+400,i,f'{int(n):,}',va='center',fontsize=7);a.set_xlim(0,36000)
    a=ax[2];a.axis('off');a.set_xlim(0,1);a.set_ylim(0,1)
    a=ax[3];scored=[11,11,15];unscored=[2,2,2]
    a.bar(range(3),scored,color=[COL[x] for x in ORDER],label='Scored')
    a.bar(range(3),unscored,bottom=scored,color='#DCE0E5',label='Below support')
    a.set_xticks(range(3),['Pulmonary','Hepatic','Cardiac']);a.set_ylabel('Annotated target types');a.set_ylim(0,20);a.legend(frameon=False,loc='upper left')
    for i,(n,u) in enumerate(zip(scored,unscored)):a.text(i,n/2,str(n),ha='center',va='center',color='white',fontweight='bold')
    a=ax[4];a.axis('off');a.set_xlim(0,1);a.set_ylim(0,1)
    j,z=anchor_row();known=z['test_y']>=0;pt=softmax(z['test_logits']/j['temperature'],axis=1).max(1)
    a=ax[5]
    a.hist(pt[known],bins=np.linspace(0,1,21),density=True,histtype='step',lw=1.5,color='#396FAC',label='Held known')
    a.hist(pt[~known],bins=np.linspace(0,1,21),density=True,histtype='stepfilled',alpha=.28,color='#BA526C',label='Held target')
    threshold=1-j['scores']['msp_temperature']['operating_points'][2]['threshold'];a.axvline(threshold,color='black',ls='--',lw=1)
    a.set_xlabel('Maximum scaled class probability');a.set_ylabel('Density');a.legend(frameon=False,loc='upper left');a.set_xlim(0,1)
    a=ax[6];grid=np.geomspace(.3,10,100);nll=[]
    for t in grid:
        p=softmax(z['calibration_logits']/t,axis=1);nll.append(-np.log(np.maximum(p[np.arange(len(p)),z['calibration_y']],1e-12)).mean())
    a.plot(grid,nll,color='#218572',lw=1.5);a.axvline(j['temperature'],color='black',ls='--',lw=.9)
    a.set_xscale('log');a.set_xlabel('Temperature');a.set_ylabel('Calibration NLL')
    a.text(.98,.92,f"T = {j['temperature']:.2f}",ha='right',transform=a.transAxes,fontsize=8)
    a=ax[7];inv=json.loads((ROOT/'experiments/SC31/input_inventory.json').read_text());counts=pd.DataFrame(inv['donor_type_counts']).T
    counts.index=[x.replace('Transformed epithelium','Transformed epi.') for x in counts.index]
    heat(a,np.log10(counts+1),vmin=0,vmax=np.log10(counts.to_numpy().max()+1),cmap='Blues',label='log10(n+1)',labelsize=6.5)
    save(fig,'1')

def figure2():
    fig,ax=plate('2',['All primary target estimates','Pooled and atlas-specific medians','Within-atlas performance distributions','Held-target support and discrimination',
                     'Conditional interval width','Selective versus forced error','Separation from shuffled labels','Below-chance targets remain visible'],['deep_type_level.csv'])
    a=ax[0];offset=0
    for atlas in ORDER:
        d=P[P.atlas==atlas].sort_values('auroc');x=np.arange(offset,offset+len(d))
        a.errorbar(x,d.auroc,yerr=[d.auroc-d.auroc_ci_lo,d.auroc_ci_hi-d.auroc],fmt='o',ms=3,lw=.6,color=COL[atlas],label=atlas)
        offset+=len(d)+2
    hchance(a);a.set_xlabel('Targets ordered within atlas');a.set_ylabel('AUROC');a.set_ylim(.05,1.02);a.set_xticks([]);a.legend(frameon=False,fontsize=6.5,loc='lower right')
    a=ax[1];vals=[P.auroc.median()]+[P[P.atlas==v].auroc.median() for v in ORDER]
    a.bar(range(4),vals,color=['#757D86']+[COL[v] for v in ORDER],width=.55);hchance(a)
    a.set_xticks(range(4),['Pooled','Pulm.','Hepatic','Cardiac']);a.set_ylim(0,1.05);a.set_ylabel('Median type AUROC')
    for i,(v,n) in enumerate(zip(vals,['7/37','1/11','6/11','0/15'])):a.text(i,v+.025,f'{v:.3f}',ha='center',fontsize=8);a.text(i,.05,n,ha='center',fontsize=7,color='white')
    a=ax[2]
    for atlas in ORDER:
        v=np.sort(P[P.atlas==atlas].auroc);a.step(v,np.arange(1,len(v)+1)/len(v),where='post',color=COL[atlas],label=atlas,lw=1.6)
    vchance(a);a.set_xlabel('Type AUROC');a.set_ylabel('Fraction of target types');a.set_xlim(0,1);a.set_ylim(0,1.03)
    for idx,col in [(3,'auroc'),(4,'width'),(5,'risk'),(6,'null')]:
        a=ax[idx]
        for atlas in ORDER:
            d=P[P.atlas==atlas]
            if col=='auroc':a.scatter(d.n_in_held_donor,d.auroc,color=COL[atlas],s=22,alpha=.8,edgecolors='white',linewidth=.4)
            elif col=='width':a.scatter(d.n_in_held_donor,d.auroc_ci_hi-d.auroc_ci_lo,color=COL[atlas],s=22,alpha=.8)
            elif col=='risk':a.scatter(1-d.acc_forced_mixed,1-d['acc@80cov_mixed'],color=COL[atlas],s=22,alpha=.85)
            else:a.scatter(d.shuffle_auroc,d.auroc,color=COL[atlas],s=22,alpha=.85)
        if col=='auroc':a.set_xscale('log');a.set_xlabel('Held-target cells');a.set_ylabel('Type AUROC');hchance(a)
        elif col=='width':a.set_xscale('log');a.set_xlabel('Held-target cells');a.set_ylabel('Cell-bootstrap interval width')
        elif col=='risk':diagonal(a,0,1);a.set_xlabel('Forced mixture error');a.set_ylabel('Error at held-mixture top 80%')
        else:diagonal(a,.1,1);a.set_xlabel('Shuffled-label AUROC');a.set_ylabel('Observed AUROC')
    a=ax[7];d=P[P.auroc<.5].sort_values('auroc');labels=[textwrap.fill(x,26) for x in d.type]
    # Explicitly smaller row labels are confined to this seven-row diagnostic.
    for i,(_,r) in enumerate(d.iterrows()):a.errorbar(r.auroc,i,xerr=[[r.auroc-r.auroc_ci_lo],[r.auroc_ci_hi-r.auroc]],fmt='o',ms=4,color=COL[r.atlas],lw=1)
    a.set_yticks(range(len(d)),labels,fontsize=6.7);a.set_xlabel('AUROC and conditional cell interval');a.set_xlim(0,.6);vchance(a);a.invert_yaxis()
    save(fig,'2')

def figure3():
    fig,ax=plate('3',['Pooled structural association','The same records retain atlas identity','Association after atlas centering','Correlation estimates and intervals',
                     'Within and between contributions','Paired change after centering','Omitting one atlas','Sensitivity to atlas mixture weights'],['deep_type_level.csv','aggregation-extension.json','omission-sensitivity.csv','atlas-weight-sensitivity.csv'])
    x=np.log10(P.nearest_centroid_euclidean.to_numpy());y=P.auroc.to_numpy();g=P.atlas.to_numpy()
    for k in [0,1,2]:
        a=ax[k]
        if k==0:
            a.scatter(x,y,color='#66717B',s=23);xx=np.linspace(x.min(),x.max(),100);a.plot(xx,np.polyval(np.polyfit(x,y,1),xx),color='#283A4A',lw=1.5)
        else:
            for atlas in ORDER:
                m=g==atlas;xx=x[m].copy();yy=y[m].copy()
                if k==2:xx-=xx.mean();yy-=yy.mean()
                a.scatter(xx,yy,s=23,color=COL[atlas],label=atlas)
                if k==1:
                    line=np.linspace(xx.min(),xx.max(),50);a.plot(line,np.polyval(np.polyfit(xx,yy,1),line),color=COL[atlas],lw=1)
            if k==1:a.legend(frameon=False,loc='lower right',fontsize=6.5)
            if k==2:a.axhline(0,color='#9AA1A9',lw=.7);a.axvline(0,color='#9AA1A9',lw=.7)
        a.set_xlabel('Centered log10 distance' if k==2 else 'log10 nearest-centroid distance');a.set_ylabel('Centered AUROC' if k==2 else 'Type AUROC')
        if k!=1:
            a.text(.02,.94,f"r = {AG['covariates']['log10_distance']['centered_r' if k==2 else 'pooled_r']:+.3f}",transform=a.transAxes,fontsize=8,va='top')
    names=['raw_distance','log10_distance','silhouette'];labels=['Raw distance','Log distance','Silhouette'];a=ax[3]
    for i,name in enumerate(names):
        d=AG['covariates'][name]
        for off,key,col,mark in [(-.13,'pooled','#283A4A','o'),(.13,'centered','#B66483','s')]:
            val=d[key+'_r'];lo,hi=d[key+'_interval'];a.errorbar(val,i+off,xerr=[[val-lo],[hi-val]],fmt=mark,ms=4,color=col,capsize=2,lw=1)
    a.axvline(0,color='#9AA1A9',ls='--',lw=.8);a.set_yticks(range(3),labels);a.set_xlim(-.65,.85);a.set_xlabel('Pearson r; 95% type-resampling interval')
    a.plot([],[],'o',color='#283A4A',label='Pooled');a.plot([],[],'s',color='#B66483',label='Centered');a.set_ylim(-.5,2.9);a.legend(frameon=False,loc='upper left',ncol=2)
    a=ax[4]
    for i,name in enumerate(names):
        d=AG['covariates'][name];cp=d['cross_product'];den=cp['total']/d['pooled_r'];vals=[cp['within']/den,cp['between']/den]
        a.barh(i-.15,vals[0],height=.27,color='#218572');a.barh(i+.15,vals[1],height=.27,color='#D69A3D')
    a.axvline(0,color='#8A939C',lw=.7);a.set_yticks(range(3),['']*3);a.set_xlabel('Contribution using pooled denominator')
    a.set_ylim(-.5,2.9);a.legend(handles=[Patch(color='#218572',label='Within'),Patch(color='#D69A3D',label='Between')],frameon=False,loc='upper left',ncol=2)
    a=ax[5]
    for i,name in enumerate(names):
        d=AG['covariates'][name];val=d['paired_difference'];lo,hi=d['paired_difference_interval']
        a.errorbar(val,i,xerr=[[val-lo],[hi-val]],fmt='o',ms=5,color='#396FAC',capsize=3,lw=1.3)
    a.set_yticks(range(3),['']*3);a.set_ylim(-.5,2.9);a.axvline(0,color='#8A939C',lw=.8,ls='--');a.set_xlabel('Pooled r minus centered r');a.set_xlim(-.2,.85)
    a=ax[6];d=OM[(OM.covariate=='log10_distance')&(OM.omission=='atlas')].set_index('omitted').loc[ORDER]
    for i,atlas in enumerate(ORDER):
        a.plot([d.loc[atlas,'pooled_r'],d.loc[atlas,'centered_r']],[i,i],color='#B2BBC3',lw=1.5)
        a.scatter(d.loc[atlas,'pooled_r'],i,color=COL[atlas],marker='o',s=32);a.scatter(d.loc[atlas,'centered_r'],i,color=COL[atlas],marker='s',s=32)
    a.axvline(0,color='#8A939C',lw=.8,ls='--');a.set_yticks(range(3),['Pulmonary omitted','Hepatic omitted','Cardiac omitted']);a.set_xlabel('r after omitting one complete atlas');a.set_xlim(-.55,.75)
    a=ax[7];xx=W.cardiac_weight+W.hepatic_weight/2;yy=np.sqrt(3)*W.hepatic_weight/2
    im=a.scatter(xx,yy,c=W.weighted_median,cmap='viridis',vmin=.4,vmax=.95,s=20,marker='h',edgecolors='none')
    a.plot([0,1,.5,0],[0,0,np.sqrt(3)/2,0],color='#53606E',lw=.7)
    a.scatter((15+11/2)/37,np.sqrt(3)*11/74,marker='*',s=90,facecolor='white',edgecolor='black',lw=.7,zorder=4)
    a.text(-.02,-.07,'Pulmonary',ha='left',fontsize=7.5);a.text(1.02,-.07,'Cardiac',ha='right',fontsize=7.5);a.text(.5,.91,'Hepatic',ha='center',fontsize=7.5)
    a.set_xlim(-.08,1.08);a.set_ylim(-.12,.98);a.axis('off');pos=a.get_position();cax=fig.add_axes([pos.x1+.012,pos.y0+.04,.012,pos.height-.07]);a._colorbar_axes=cax;fig.colorbar(im,cax=cax).ax.tick_params(labelsize=6.5,length=2)
    save(fig,'3')

def figure4():
    fig,ax=plate('4',['Pulmonary donor assignments','Hepatic donor assignments','Adult-cardiac donor assignments','Each held donor has its own summary',
                     'Target instability across assignments','Sensitivity to readout specification','Agreement with the baseline decision','Selective risk across existing assignments'],['historical-all-donor-pairs.csv','historical-specifications.csv'])
    for i,atlas in enumerate(ORDER):
        d=DS[DS.atlas==atlas];frame=d.pivot_table(index='test_donor',columns='calibration_donor',values='auroc',aggfunc='median')
        short=lambda v:v.replace('ASK','').replace('_P1TLH','').replace('_P2TLH','').replace('_P3TLH','').replace('_P4TLH','').replace('_P5TLH','')
        frame.index=[short(v) for v in frame.index];frame.columns=[short(v) for v in frame.columns]
        heat(ax[i],frame,cmap='viridis',fmt='.2f',label='Median',labelsize=6.8,compact_numbers=True)
        ax[i].set_xlabel('Calibration donor');ax[i].set_ylabel('Held donor')
    a=ax[3]
    for j,atlas in enumerate(ORDER):
        s=DS[DS.atlas==atlas].groupby('test_donor').auroc.median().sort_values();positions=np.linspace(j-.17,j+.17,len(s))
        a.scatter(positions,s,s=28,color=COL[atlas],zorder=3)
        a.plot([j-.25,j+.25],[s.median(),s.median()],color='black',lw=1.6)
    a.set_xticks(range(3),['Pulmonary','Hepatic','Cardiac']);a.set_ylabel('Median across eligible tasks');a.set_ylim(.35,1);hchance(a)
    a=ax[4];spread=DS.groupby(['atlas','type']).auroc.agg(['min','max']).reset_index();spread['range']=spread['max']-spread['min']
    distribution(a,spread,'range','atlas',ORDER,COL,labels=['Pulmonary','Hepatic','Cardiac'],limits=(0,1));a.set_ylabel('Per-target AUROC range')
    for i,atlas in enumerate(ORDER):
        d=spread[spread.atlas==atlas];n=sum((d['min']<.5)&(d['max']>.5));a.text(i,.96,f'{n}/{len(d)} cross 0.5',ha='center',fontsize=7)
    specs=['hvg500_pca30_logistic','hvg2000_pca30_logistic','hvg1000_pca20_logistic','hvg1000_pca50_logistic','hvg1000_pca30_sgd_log_loss']
    spl=['HVG\n500','HVG\n2000','PC\n20','PC\n50','SGD']
    base=S[(S.specification_id=='baseline_hvg1000_pca30_logistic')&(S.status=='scored')]
    matched=S[(S.specification_id!='baseline_hvg1000_pca30_logistic')&(S.status=='scored')].merge(base,on=['atlas','type','test_donor','calibration_donor'],suffixes=('','_baseline'))
    matched['delta']=matched.auroc-matched.auroc_baseline
    a=ax[5]
    for i,s in enumerate(specs):
        for j,atlas in enumerate(ORDER):
            v=matched[(matched.specification_id==s)&(matched.atlas==atlas)].delta;lo,med,hi=np.quantile(v,[.25,.5,.75]);x=i+(j-1)*.22
            a.errorbar(x,med,yerr=[[med-lo],[hi-med]],fmt='o',color=COL[atlas],ms=3.5,capsize=2,lw=1)
    a.axhline(0,color='#8A939C',lw=.8);a.set_xticks(range(5),spl);a.set_ylabel('AUROC change; median and IQR')
    a=ax[6]
    for j,atlas in enumerate(ORDER):
        vals=[]
        for s in specs:
            d=matched[(matched.specification_id==s)&(matched.atlas==atlas)];vals.append(np.mean((d.ci_low>.5)==(d.ci_low_baseline>.5)))
        a.plot(range(5),vals,'-o',color=COL[atlas],ms=3,label=atlas,lw=1)
    a.set_xticks(range(5),spl);a.set_ylim(0,1.04);a.set_ylabel('Conditional detection agreement');a.legend(frameon=False,loc='lower left',fontsize=6.5)
    a=ax[7];data=DS.melt(id_vars=['atlas'],value_vars=['risk_0.60','risk_0.70','risk_0.80','risk_0.90'],var_name='q',value_name='risk');data['coverage']=data.q.str.replace('risk_','').astype(float)
    curves(a,data,'coverage','risk','atlas',ORDER,COL);a.set_xlabel('Requested held-mixture coverage');a.set_ylabel('Selective risk; median and IQR');a.set_ylim(0,.75)
    save(fig,'4')

def figure5():
    fig,ax=plate('5',['Matched per-cell novelty scores','Mean target ROC on the anchor split','Precision relative to target prevalence','Target-by-score interaction',
                     'Donor-by-score interaction','Effect of temperature on AUROC','Shuffling training labels','Fresh predictions versus frozen primary'],['new-score-comparisons.csv','new-label-controls.csv','SC31/scores','deep_type_level.csv'])
    a=ax[0];distribution(a,N,'auroc','method',METHODS,MC,labels=MSHORT,limits=(0,1.04));a.set_ylabel('AUROC across 120 evaluations');a.tick_params(axis='x',labelsize=7);hchance(a)
    a=ax[1];grid=np.linspace(0,1,201);acc={m:[] for m in METHODS}
    anchors=N[(N.test_donor=='ASK440')&(N.calibration_donor=='ASK452')&(N.method=='msp_temperature')]
    for _,row in anchors.iterrows():
        z=np.load(ROOT/'experiments/SC31/scores'/f'{row.row_id}.npz');j=json.loads((ROOT/'experiments/SC31/rows'/f'{row.row_id}.json').read_text())
        from run_prediction_extension import get_scores
        scores=get_scores(z['test_logits'],z['test_distance'],j['temperature'])
        for m,s in scores.items():
            fpr,tpr,_=roc_curve(z['test_y']==-1,s);acc[m].append(np.interp(grid,fpr,tpr))
    for m in METHODS:a.plot(grid,np.mean(acc[m],axis=0),color=MC[m],label=MN[m],lw=1.2)
    diagonal(a);a.set_xlabel('False positive rate');a.set_ylabel('Mean true positive rate');a.legend(frameon=False,loc='lower right',fontsize=6.2)
    a=ax[2];en=N.copy();en['enrichment']=en.average_precision/en.novel_prevalence
    distribution(a,en,'enrichment','method',METHODS,MC,labels=MSHORT);a.tick_params(axis='x',labelsize=7);a.set_yscale('log');a.set_ylabel('Average precision / prevalence');a.axhline(1,color='#8A939C',ls='--',lw=.8)
    a=ax[3];frame=N.pivot_table(index='type',columns='method',values='auroc',aggfunc='median')[METHODS];frame.columns=[x.replace('\n',' ') for x in MSHORT]
    frame.index=[x.replace('Transformed epithelium','Transformed epi.') for x in frame.index];heat(a,frame,label='Median',labelsize=6.5);a.tick_params(axis='x',rotation=35)
    for tick in a.get_xticklabels():tick.set_ha('right');tick.set_rotation_mode('anchor')
    a=ax[4];frame=N.pivot_table(index='test_donor',columns='method',values='auroc',aggfunc='median')[METHODS];frame.columns=[x.replace('\n',' ') for x in MSHORT];heat(a,frame,vmin=.4,vmax=.9,fmt='.2f',label='Median',labelsize=6.5);a.tick_params(axis='x',rotation=35)
    for tick in a.get_xticklabels():tick.set_rotation(90);tick.set_ha('center');tick.set_rotation_mode('default')
    a=ax[5];wide=N.pivot(index='row_id',columns='method',values='auroc');a.scatter(wide.msp_raw,wide.msp_temperature,s=15,color='#218572',alpha=.7,label='MSP');a.scatter(wide.entropy_raw,wide.entropy_temperature,s=15,color='#6F55A5',alpha=.65,label='Entropy');diagonal(a);a.set_xlabel('AUROC without scaling');a.set_ylabel('AUROC with temperature scaling');a.legend(frameon=False,loc='lower right')
    a=ax[6];d=NC[NC.method=='msp_temperature'];a.scatter(d.observed_auroc,d.control_auroc,color='#B94E5D',s=28,edgecolor='white',lw=.4);diagonal(a);hchance(a);a.set_xlabel('Observed readout AUROC');a.set_ylabel('Within-donor label-shuffle AUROC')
    a=ax[7];d=anchors.merge(P[P.atlas=='Pulmonary'],on='type',suffixes=('_fresh','_primary'));a.scatter(d.auroc_primary,d.auroc_fresh,s=30,color='#218572');diagonal(a,.25,1);a.set_xlabel('Frozen primary AUROC');a.set_ylabel('Fresh anchor AUROC');a.text(.03,.92,f"Maximum absolute change\n{np.abs(d.auroc_primary-d.auroc_fresh).max():.5f}",transform=a.transAxes,va='top',fontsize=8)
    save(fig,'5')

def figure6():
    fig,ax=plate('6',['Known-cell retention after transfer','Novel cells accepted by the threshold','Actual coverage and retained error','Discrimination and novel-cell acceptance',
                     'Known-class calibration error','Held-donor negative log likelihood','Effect of novel-cell prevalence','Selection improves some tasks more than others'],['new-locked-thresholds.csv','new-known-calibration.csv','new-prevalence-sensitivity.csv','new-score-comparisons.csv'])
    for idx,metric,ylabel in [(0,'held_known_coverage','Held-known retained fraction'),(1,'novel_acceptance','Held-target accepted fraction')]:
        a=ax[idx];curves(a,O,'nominal_known_coverage',metric,'method',METHODS,MC,MN);a.set_xlabel('Nominal calibration-known coverage');a.set_ylabel(ylabel);a.set_ylim(0,1.02)
        if idx==0:a.plot([.6,.95],[.6,.95],color='#8A939C',lw=.8,ls='--')
        else:a.legend(frameon=False,loc='lower right',fontsize=6.3)
    q=O[np.isclose(O.nominal_known_coverage,.8)]
    a=ax[2]
    for m in METHODS:
        d=q[q.method==m];a.scatter(d.held_mixture_coverage,d.selective_risk,s=9,color=MC[m],alpha=.37)
        a.scatter(d.held_mixture_coverage.median(),d.selective_risk.median(),s=55,color=MC[m],edgecolor='black',linewidth=.7,zorder=4)
    a.set_xlim(0,1.02);a.set_ylim(0,.9);a.set_xlabel('Realized held-mixture coverage');a.set_ylabel('Error among accepted cells')
    a=ax[3];d=q[q.method=='msp_temperature'].merge(N[N.method=='msp_temperature'][['row_id','auroc']],on='row_id')
    colorset=dict(zip(sorted(d.test_donor.unique()),['#218572','#396FAC','#B66483','#C28B29']))
    for donor,dd in d.groupby('test_donor'):a.scatter(dd.auroc,dd.novel_acceptance,s=18,color=colorset[donor],alpha=.8,label=donor)
    a.set_xlim(0,1);a.set_ylim(0,1.02);a.set_xlabel('Target-versus-known AUROC');a.set_ylabel('Novel acceptance at 80%');a.legend(frameon=False,ncol=2,loc='lower left',fontsize=6.3)
    for idx,metric,label in [(4,'ece','ECE'),(5,'nll','NLL')]:
        a=ax[idx];w=C.pivot(index='row_id',columns='calibration',values=metric);a.scatter(w.raw,w.temperature,s=15,color='#218572',alpha=.6)
        hi=max(w.raw.max(),w.temperature.max())*1.04;diagonal(a,0,hi);a.set_xlabel(label+' before scaling');a.set_ylabel(label+' after scaling')
    a=ax[6];curves(a,PV[np.isclose(PV.nominal_known_coverage,.8)],'novel_prevalence','risk','method',METHODS,MC,MN);a.set_xlabel('Hypothetical novel-cell prevalence');a.set_ylabel('Risk at fixed calibration threshold');a.set_ylim(0,1)
    a=ax[7];d=q[q.method=='msp_temperature'];a.scatter(d.forced_risk,d.selective_risk,s=20,color='#218572',alpha=.7);diagonal(a,0,1);a.set_xlabel('Forced annotation error');a.set_ylabel('Error at calibrated cutoff')
    save(fig,'6')

def figure7():
    fig,ax=plate('7',['A monotone change preserves score ranking','ROC curves remain identical','Known-cell retention under score shift','Novel-cell acceptance under score shift',
                     'Selective risk under score shift','Discrimination stays fixed','Prevalence and requested retention','Donor differences in threshold transfer'],['rank-shift-stress.csv','new-prevalence-sensitivity.csv','SC31/scores'])
    a=ax[0];x=np.linspace(0,1,201)
    for delta,color in [(-2,'#396FAC'),(0,'#65707A'),(2,'#BA526C')]:
        f=np.exp(delta);a.plot(x,f*x/(1-x+f*x),label=f'Shift {delta:+d}',color=color,lw=1.5)
    a.set_xlabel('Original novelty score');a.set_ylabel('Transformed novelty score');a.legend(frameon=False,loc='lower right')
    j,z=anchor_row();s=1-softmax(z['test_logits']/j['temperature'],axis=1).max(1);a=ax[1]
    for delta,color,style in [(-2,'#396FAC','-'),(0,'#65707A','--'),(2,'#BA526C',':')]:
        f=np.exp(delta);t=f*s/(1-s+f*s);fpr,tpr,_=roc_curve(z['test_y']==-1,t);a.plot(fpr,tpr,color=color,lw=1.8,ls=style,label=f'{delta:+d}')
    diagonal(a);a.set_xlabel('False positive rate');a.set_ylabel('True positive rate');a.legend(frameon=False,title='Held score shift',loc='lower right',fontsize=7,title_fontsize=7)
    for k,col,lab in [(2,'known_coverage','Known-cell retained fraction'),(3,'novel_acceptance','Novel-cell accepted fraction'),(4,'risk','Error among accepted cells')]:
        a=ax[k];d=ST.groupby('shift')[col].quantile([.25,.5,.75]).unstack();a.plot(d.index,d[.5],'-o',ms=4,color='#218572',lw=1.7);a.fill_between(d.index,d[.25],d[.75],color='#218572',alpha=.17,lw=0)
        a.axvline(0,color='#8A939C',ls='--',lw=.7);a.set_ylim(0,1.03);a.set_xlabel('Held-score log-odds shift');a.set_ylabel(lab)
    a=ax[5];d=ST.groupby('shift').auroc.quantile([.25,.5,.75]).unstack();a.plot(d.index,d[.5],'-o',ms=4,color='#396FAC');a.fill_between(d.index,d[.25],d[.75],color='#396FAC',alpha=.17,lw=0);a.set_ylim(0,1);a.set_xlabel('Held-score log-odds shift');a.set_ylabel('AUROC; median and IQR')
    a.text(.02,.05,f"Maximum per-task change\n= {ST.auroc_difference.abs().max():.1f}",transform=a.transAxes,fontsize=8)
    a=ax[6];d=PV[PV.method=='msp_temperature'].pivot_table(index='nominal_known_coverage',columns='novel_prevalence',values='risk',aggfunc='median');d.index=[f'{v:.0%}' for v in d.index];d.columns=[f'{v:.0%}' for v in d.columns];heat(a,d,vmin=0,vmax=1,cmap='magma',fmt='.2f',label='Risk',colorbar_title_pad=9);a.set_xlabel('Novel-cell prevalence');a.set_ylabel('Calibration-known coverage')
    a=ax[7];donors=sorted(ST.test_donor.unique());cols=dict(zip(donors,['#218572','#396FAC','#B66483','#C28B29']))
    for donor in donors:
        d=ST[ST.test_donor==donor].groupby('shift').known_coverage.median();a.plot(d.index,d,'-o',ms=3,color=cols[donor],label=donor,lw=1.2)
    a.set_xlabel('Held-score log-odds shift');a.set_ylabel('Median held-known retained fraction');a.set_ylim(0,1.02);a.legend(frameon=False,fontsize=9.8,loc='lower left',ncol=2,columnspacing=.7,handlelength=1.0,handletextpad=.4)
    save(fig,'7')

def figure8():
    fig,ax=plate('8',['Target-level frozen-representation results','Distribution across the same target set','Selective risk at fixed requested coverage','Diagnostic score error and calibration',
                     'Targets below chance across configurations','Fitted temperatures','Agreement of target rankings','Representation spread within each target'],['historical-frozen-representations.csv'])
    a=ax[0];frame=FM.pivot(index='type',columns='model',values='auroc')[MFO];frame.columns=MFL;frame.index=[x.replace('Transformed epithelium','Transformed epi.') for x in frame.index];heat(a,frame,label='AUROC',labelsize=6.5);a.tick_params(axis='x',rotation=90)
    for tick in a.get_xticklabels():tick.set_ha('center');tick.set_rotation_mode('default')
    cmap=dict(zip(MFO,['#396FAC','#7B9DCE','#218572','#C28B29','#BA526C','#6F55A5']))
    a=ax[1];distribution(a,FM,'auroc','model',MFO,cmap,labels=MFL,limits=(0,1.05));a.set_ylabel('AUROC across 11 targets');a.tick_params(axis='x',rotation=90)
    for tick in a.get_xticklabels():tick.set_ha('center');tick.set_rotation_mode('default')
    hchance(a)
    a=ax[2];data=FM.melt(id_vars=['model'],value_vars=['risk_0.60','risk_0.70','risk_0.80','risk_0.90'],var_name='q',value_name='risk');data['coverage']=data.q.str.replace('risk_','').astype(float)
    curves(a,data,'coverage','risk','model',MFO,cmap,dict(zip(MFO,MFL)),band=False);a.set_xlabel('Requested held-mixture coverage');a.set_ylabel('Median selective risk');a.set_ylim(-.003,.22);a.legend(frameon=False,ncol=2,fontsize=6.2,loc='upper left')
    a=ax[3]
    for m in MFO:
        d=FM[FM.model==m];a.scatter(d.brier,d.ece,s=18,color=cmap[m],alpha=.8)
    a.set_xlabel('Diagnostic novelty-score Brier value');a.set_ylabel('Diagnostic expected calibration error')
    a=ax[4];counts=FM.assign(below=FM.auroc<.5).groupby('type').below.sum().sort_values();a.barh(range(len(counts)),counts,color='#BA526C');a.set_yticks(range(len(counts)),[v.replace('Transformed epithelium','Transformed epi.') for v in counts.index],fontsize=6.7);a.set_xlabel('Configurations with AUROC below 0.5');a.set_xlim(0,6.3);a.set_xticks([0,2,4,6])
    a=ax[5];distribution(a,FM,'temperature','model',MFO,cmap,labels=MFL);a.set_ylabel('Calibration temperature');a.tick_params(axis='x',rotation=90)
    for tick in a.get_xticklabels():tick.set_ha('center');tick.set_rotation_mode('default')
    a.axhline(1,color='#8A939C',ls='--',lw=.8)
    a=ax[6];frame=FM.pivot(index='type',columns='model',values='auroc')[MFO].corr(method='spearman');frame.index=MFL;frame.columns=MFL;heat(a,frame,vmin=-1,vmax=1,cmap='RdBu_r',fmt='.2f',label='Rank r',labelsize=6.5,adaptive_contrast=True,horizontal_colorbar=True);a.tick_params(axis='x',rotation=90)
    for tick in a.get_xticklabels():tick.set_ha('center');tick.set_rotation_mode('default')
    a=ax[7];d=FM.groupby('type').auroc.agg(['min','median','max']).sort_values('median');y=np.arange(len(d));a.hlines(y,d['min'],d['max'],color='#778999',lw=2);a.scatter(d['median'],y,color='#396FAC',s=18);a.set_yticks(y,[v.replace('Transformed epithelium','Transformed epi.') for v in d.index],fontsize=6.7);a.set_xlabel('Across-configuration AUROC range');a.set_xlim(0,1.03);vchance(a)
    save(fig,'8')

def figure9():
    datasets=['gastric','ca_cancer','lung_adre'];display={'gastric':'Gastric context','ca_cancer':'Carcinoma context','lung_adre':'Pulmonary context'}
    models=['Geneformer-316M','scGPT','UCE'];ml=['GF large','scGPT','UCE'];cc={'gastric':'#218572','ca_cancer':'#BA526C','lung_adre':'#396FAC'}
    fig,ax=plate('9',['Gastric cached context','Carcinoma cached context','Pulmonary cached context','Every combination retains its target spread',
                     'Selective risk in cached contexts','Discrimination and diagnostic calibration','Eligible targets and held-group support','Matched representation contrasts'],['historical-portability.csv'])
    for i,dataset in enumerate(datasets):
        d=EX[EX.dataset==dataset];frame=d.pivot(index='type',columns='model',values='auroc')[models];frame.columns=ml;frame.index=[textwrap.shorten(str(v),width=24,placeholder='...') for v in frame.index]
        heat(ax[i],frame,label='AUROC',fmt='.2f',labelsize=6.8,scale_heading_above_frame=True)
        if i<2:ax[i]._colorbar_axes.title.set_x(-.8)
    a=ax[3];y=0;yt=[];yl=[]
    for ds in datasets:
        for m,l in zip(models,ml):
            v=EX[(EX.dataset==ds)&(EX.model==m)].auroc;a.hlines(y,v.min(),v.max(),color=cc[ds],lw=2);a.scatter(v.median(),y,color=cc[ds],s=22);yt.append(y);yl.append(ds.replace('_',' ')+' / '+l);y+=1
    a.set_yticks(yt,yl,fontsize=6.5);a.set_xlim(0,1.03);vchance(a);a.set_xlabel('Median and full target AUROC range');a.invert_yaxis()
    a=ax[4]
    for ds in datasets:
        for m,ls in zip(models,['-','--',':']):
            d=EX[(EX.dataset==ds)&(EX.model==m)];a.plot([.6,.7,.8,.9],[d[f'risk_{q:.2f}'].median() for q in [.6,.7,.8,.9]],color=cc[ds],ls=ls,lw=1.3,marker='o',ms=2.5)
    a.set_xlabel('Requested held-mixture coverage');a.set_ylabel('Median selective risk');a.set_ylim(0,.5)
    for m,ls in zip(ml,['-','--',':']):a.plot([],[],ls=ls,color='#56616D',label=m)
    a.legend(frameon=False,fontsize=6.6,loc='upper left')
    a=ax[5]
    for ds in datasets:
        d=EX[EX.dataset==ds];a.scatter(d.auroc,d.ece,color=cc[ds],s=18,alpha=.65,label=display[ds])
    a.set_xlabel('Target AUROC');a.set_ylabel('Diagnostic calibration error');a.set_xlim(0,1);a.legend(frameon=False,loc='upper left',fontsize=6.2)
    a=ax[6]
    for i,ds in enumerate(datasets):
        d=EX[(EX.dataset==ds)&(EX.model==models[0])];a.scatter(np.repeat(i,len(d)),d.n_target_in_held,s=28,color=cc[ds],alpha=.7);a.text(i,d.n_target_in_held.max()*1.28,f'{len(d)} targets',ha='center',fontsize=7)
    a.set_xticks(range(3),['Gastric','Carcinoma','Pulmonary']);a.set_xlim(-.4,2.4);a.set_yscale('log');a.set_ylabel('Held-target cells');a.set_ylim(20,EX.n_target_in_held.max()*2)
    a=ax[7];labels=['scGPT - GF','UCE - GF','UCE - scGPT']
    for i,(l,r) in enumerate([('scGPT','Geneformer-316M'),('UCE','Geneformer-316M'),('UCE','scGPT')]):
        for j,ds in enumerate(datasets):
            d=EX[EX.dataset==ds].pivot(index='type',columns='model',values='auroc');delta=d[l]-d[r];x=i+(j-1)*.2;lo,med,hi=np.quantile(delta,[.25,.5,.75]);a.errorbar(x,med,yerr=[[med-lo],[hi-med]],fmt='o',color=cc[ds],ms=4,capsize=2)
    a.axhline(0,color='#8A939C',ls='--',lw=.8);a.set_xticks(range(3),['scGPT\n− GF','UCE\n− GF','UCE\n− scGPT'],fontsize=9.8);a.set_ylabel('Paired AUROC difference')
    save(fig,'9')

def supplementary1():
    fig=plt.figure(figsize=(8.4,10.25));panels=[]
    for i,atlas in enumerate(ORDER):
        y0=[.725,.415,.055][i];height=[.225,.225,.27][i]
        a=fig.add_axes([.29,y0,.60,height]);d=P[P.atlas==atlas].sort_values('auroc');yy=np.arange(len(d))
        a.errorbar(d.auroc,yy,xerr=[d.auroc-d.auroc_ci_lo,d.auroc_ci_hi-d.auroc],fmt='o',color=COL[atlas],ms=4,capsize=2,lw=1)
        a.set_yticks(yy,[textwrap.fill(v,35) for v in d.type],fontsize=7.7);a.invert_yaxis();a.set_xlim(0,1.02);vchance(a);a.set_xlabel('Type AUROC and conditional cell-bootstrap interval')
        a.set_title(atlas,loc='center',fontweight='normal',pad=7,fontsize=11.8)
        a._panel_letter=fig.text(.24,y0+height+.015,chr(65+i),fontweight='bold',fontsize=15)
        for j,n in enumerate(d.n_in_held_donor):a.text(1.05,j,f'{n:,}',va='center',fontsize=7.5,clip_on=False)
        a.text(1.05,-1.4,'Held n',fontsize=7.5,clip_on=False)
        panels.append({'panel':chr(65+i),'title':atlas+' complete target intervals','sources':['deep_type_level.csv'],
                       'axes_rect':[.29,y0,.60,height]})
    MANIFEST.append({'figure':'S1','layout':'hand-positioned complete target forests',
                     'figure_inches':[8.4,10.25],'panels':panels});save(fig,'S1')

def supplementary2():
    targets=['Ciliated','Macrophages','Type 2']
    titles=['Ciliated known-class calibration','Ciliated score distributions','Macrophage known-class calibration','Macrophage score distributions',
            'Type 2 known-class calibration','Type 2 score distributions','Temperature across held donors','Where withheld targets are assigned']
    fig,ax=plate('S2',titles,['SC31/rows','SC31/scores','new-score-comparisons.csv'])
    for i,target in enumerate(targets):
        d,z=anchor_row(target);a=ax[i*2]
        for state,color,label in [('raw','#7C8791','Unscaled'),('temperature','#218572','Scaled')]:
            b=d['known_test_calibration_'+state]['bins'];b=[x for x in b if x['n']>0]
            a.plot([x['confidence'] for x in b],[x['accuracy'] for x in b],'-o',color=color,lw=1,ms=3,label=label)
        diagonal(a);a.set_xlabel('Mean predicted confidence in bin');a.set_ylabel('Observed known-class accuracy');a.legend(frameon=False,loc='lower right')
        a=ax[i*2+1];cs=1-softmax(z['calibration_logits']/d['temperature'],axis=1).max(1);ts=1-softmax(z['test_logits']/d['temperature'],axis=1).max(1)
        for values,color,label in [(cs,'#C28B29','Cal known'),(ts[z['test_y']>=0],'#396FAC','Held known'),(ts[z['test_y']==-1],'#BA526C','Held target')]:
            values=np.sort(values);a.plot(values,np.arange(1,len(values)+1)/len(values),color=color,lw=1.25,label=label)
        a.axvline(np.quantile(cs,.8,method='higher'),color='black',ls='--',lw=.8);a.set_xlabel('Temperature-scaled novelty score');a.set_ylabel('Empirical cumulative fraction');a.set_xlim(0,1);a.set_ylim(0,1.03);a.legend(frameon=False,loc='lower right',fontsize=6.5)
    a=ax[6];d=N[N.method=='msp_temperature'];donors=sorted(d.test_donor.unique());cmap=dict(zip(donors,['#218572','#396FAC','#B66483','#C28B29']));distribution(a,d,'temperature','test_donor',donors,cmap);a.set_ylabel('Temperature fitted on known calibration cells');a.axhline(1,color='#8A939C',ls='--',lw=.8)
    summary=[]
    for _,r in N[(N.method=='msp_temperature')&(N.test_donor=='ASK440')&(N.calibration_donor=='ASK452')].iterrows():
        d,z=anchor_row(r.type);pred=z['test_logits'][z['test_y']==-1].argmax(1);counts=np.bincount(pred,minlength=len(d['training_classes']));best=int(counts.argmax())
        summary.append({'target':r.type,'dominant_predicted_reference':d['training_classes'][best],'fraction':float(counts[best]/counts.sum()),'n_target':int(counts.sum())})
    df=pd.DataFrame(summary).sort_values('fraction');df.to_csv(E/'new-target-assignment-destinations.csv',index=False)
    a=ax[7];a.barh(range(len(df)),df.fraction,color='#396FAC',height=.6);a.set_yticks(range(len(df)),[t.replace('Transformed epithelium','Transformed epi.') for t in df.target],fontsize=6.7);a.set_xlim(0,1);a.set_xlabel('Fraction in the most frequent reference class')
    save(fig,'S2')

def supplementary3():
    primary=HC['sc31_fetal_lung_pns_novel'];rt={r['check']:r for r in HC['sc31_fetal_lung_pns_redteam']['checks']};ct={r['check']:r for r in HC['sc31_atlas_confound_control']['checks']}
    fig,ax=plate('S3',['Cross-atlas signal and provenance probe','Reference-donor assignments','Seed sensitivity','Feature-count sensitivity',
                      'Depth diagnostics','Depth matching and transductive integration','Query-type components','Forced and selective annotation error'],['historical-cross-atlas-controls.json'])
    def forest(a,labels,values,intervals):
        for i,(v,(lo,hi)) in enumerate(zip(values,intervals)):a.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color='#396FAC',capsize=2,ms=4,lw=1.2)
        a.set_yticks(range(len(labels)),labels,fontsize=7);a.set_xlim(0,1.03);a.set_ylim(-.5,len(labels)-.5);vchance(a);a.set_xlabel('AUROC and conditional cell interval')
    forest(ax[0],['Novelty','Label shuffle','Atlas identity'],[primary['novel_auroc'],primary['shuffle_auroc'],ct['atlas_label_probe']['auroc']],
           [primary['novel_auroc_ci'],primary['shuffle_auroc_ci'],ct['atlas_label_probe']['auroc_ci']])
    runs=rt['reference_donor_swap']['runs'];forest(ax[1],[v['held_gse_donor'] for v in runs],[v['novel_auroc'] for v in runs],[v['novel_auroc_ci'] for v in runs])
    for idx,key,xkey,label in [(2,'seed_sweep','seed','Seed index (five fixed seeds)'),(3,'hvg_count_sensitivity','n_hvg','Selected variable genes')]:
        a=ax[idx];rr=rt[key]['runs'];x=np.arange(len(rr)) if idx==2 else np.array([r[xkey] for r in rr]);v=np.array([r['novel_auroc'] for r in rr]);ci=np.array([r['novel_auroc_ci'] for r in rr]);a.errorbar(x,v,yerr=[v-ci[:,0],ci[:,1]-v],fmt='o-',color='#218572',capsize=3,lw=1.2,ms=4);a.set_xlabel(label);a.set_ylabel('AUROC');a.set_ylim(.5,1.02)
        if idx==2:a.set_xticks(x,[str(k+1) for k in x])
    dep=rt['depth_confound'];keys=['depth_log_ncounts','depth_log_ngenes','depth_controlled_residual','feature_regressout']
    forest(ax[4],['Library size','Detected genes','Score residual','Feature residual'],[dep[k+'_auroc'] for k in keys],[dep[k+'_ci'] for k in keys])
    matched=ct['depth_matched_subsample']['quantile_bin_overlap'];harm=ct['harmony_atlas_then_novel']
    forest(ax[5],['Original','Depth-matched','Harmony*'],[primary['novel_auroc'],matched['auroc'],harm['harmony_novel_auroc']],
           [primary['novel_auroc_ci'],matched['auroc_ci'],harm['harmony_novel_auroc_ci']])
    ax[5].text(.04,.02,'* Integration includes\nheld cells',transform=ax[5].transAxes,fontsize=7,color='#804B57')
    types=rt['type_collapse_probe']['per_type'];forest(ax[6],['Schwann precursor','Neuron'],[r['novel_auroc'] for r in types],[r['novel_auroc_ci'] for r in types])
    a=ax[7];vals=[1-primary['acc_full_forced_mixed'],1-primary['acc@80cov_mixed'],1-primary['acc_full_known_only'],1-primary['acc@80cov_known_only']]
    a.bar(range(4),vals,color=['#87939F','#396FAC','#B7C9BF','#218572']);a.set_xticks(range(4),['M\nforced','M\n80%','K\nforced','K\n80%']);a.set_ylabel('Annotation error');a.set_ylim(0,.45)
    save(fig,'S3')

if __name__=='__main__':
    for fn in [figure1,figure2,figure3,figure4,figure5,figure6,figure7,figure8,figure9,supplementary1,supplementary2,supplementary3]:
        fn();print(fn.__name__,flush=True)
    for item in MANIFEST:
        number=item['figure'];item['files']={ext:{'path':f'figures/Figure{number}.{ext}','sha256':hashlib.sha256((F/f'Figure{number}.{ext}').read_bytes()).hexdigest()} for ext in ['pdf','svg','png']}
    (E/'figure-panel-manifest.json').write_text(json.dumps({'figures':MANIFEST,'main_panels':sum(len(f['panels']) for f in MANIFEST if not str(f['figure']).startswith('S')),'supplement_panels':sum(len(f['panels']) for f in MANIFEST if str(f['figure']).startswith('S'))},indent=2)+'\n')
    (E/'figure-legend-layout-checks.json').write_text(json.dumps({'status':'PASS','figures':LAYOUT_CHECKS},indent=2)+'\n')
