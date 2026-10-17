"""Original vector schematics; source SVG is embedded into the figure SVG.

Inkscape creates the production PDF derivative. Text remains live and Arial;
all marks are manually drawn and encode donor roles, fitted operations or scores.
"""
from pathlib import Path
from xml.etree import ElementTree as ET
import copy,subprocess,sys,os,shutil
NS='http://www.w3.org/2000/svg';ET.register_namespace('',NS)
INK='#263B4A';TEAL='#218572';AMBER='#B98428';BLUE='#396FAC';ROSE='#BA526C'

def text(x,y,s,size=11,anchor='middle',color=INK):
    from xml.sax.saxutils import escape
    return f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" font-weight="normal" text-anchor="{anchor}" fill="{color}">{escape(s)}</text>'
def line(x,y,u,v,c=INK,width=1.2,dash=''):
    return f'<path d="M{x},{y} L{u},{v}" fill="none" stroke="{c}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'
def circle(x,y,r,c,fill='none',width=1.3):
    alpha=f' fill-opacity="{int(fill[7:9],16)/255:.4f}"' if fill.startswith('#') and len(fill)==9 else ''
    fill=fill[:7] if alpha else fill
    return f'<circle cx="{x}" cy="{y}" r="{r}" stroke="{c}" fill="{fill}"{alpha} stroke-width="{width}"/>'
def cell(x,y,c,r=5):return circle(x,y,r,c,c+'24')+circle(x-1,y,1.5,c,c,.7)
def person(x,y,c):
    return circle(x,y-10,5,c,c+'18')+f'<path d="M{x-10},{y+12} Q{x-9},{y-1} {x},{y-1} Q{x+9},{y-1} {x+10},{y+12}" stroke="{c}" fill="none" stroke-width="1.6" stroke-linecap="round"/>'
def star(x,y,c):
    pts=[(0,-6),(1.8,-2),(6,-1.6),(3,1.5),(3.7,5.8),(0,3.7),(-3.7,5.8),(-3,1.5),(-6,-1.6),(-1.8,-2)]
    return '<polygon points="'+' '.join(f'{x+a},{y+b}' for a,b in pts)+f'" fill="{c}" stroke="{c}" stroke-width=".5"/>'
def wrap(w,h,body):
    return f'<svg xmlns="{NS}" width="{w}pt" height="{h}pt" viewBox="0 0 {w} {h}"><g stroke-linejoin="round" stroke-linecap="round">{body}</g></svg>'
def donor():
    s=''
    for x,c,title,desc in [(86,TEAL,'Training donors','Known reference types'),(269,AMBER,'Calibration donor','Known reference types'),(452,BLUE,'Held donor','Known + withheld target')]:
        s+=person(x-49,28,c)
        for dx,dy in [(-12,-8),(3,-9),(-8,6),(9,5)]:s+=cell(x+23+dx,24+dy,c)
        if x==452:s+=star(x+47,25,ROSE)
        s+=text(x,57,title,12.5)+text(x,73,desc,10.8)
    s+=line(177,3,177,76,'#C8D1D8',.8,'2 4')+line(360,3,360,76,'#C8D1D8',.8,'2 4')
    return wrap(538,80,s)
def fitting():
    s=''
    # A shared feature transform and readout are learned only from training.
    for x,c,title,desc in [(58,TEAL,'Feature transform','HVG / scale / PCA'),(208,TEAL,'Class readout','Known training labels')]:
        s+=circle(x,22,18,c,c+'0E')
        if x==58:
            for dx,dy in [(-8,-6),(1,-7),(-7,3),(2,4)]:s+=circle(x+dx,22+dy,1.8,c,c,.5)
            s+=line(x+8,12,x+8,31,c)+line(x+4,27,x+8,31,c)+line(x+12,27,x+8,31,c)
        else:
            for dx,dy in [(-8,-7),(-7,6),(7,-5),(8,7)]:s+=circle(x+dx,22+dy,2,c,c,.5)
            s+=line(x,10,x,33,c)
        s+=text(x,54,title,11.7)+text(x,68,desc,11.4)
    s+=line(84,22,181,22,'#8296A3',1.2)+f'<path d="M176 18 L182 22 L176 26" fill="none" stroke="#8296A3" stroke-width="1.2"/>'
    # Independently held roles are distinct operations, not a donor-to-donor chain.
    s+=line(5,79,261,79,'#CFD7DD',.7)
    x=58;s+=circle(x,103,16,AMBER,AMBER+'0D')
    s+=f'<path d="M{x-9} 110 Q{x-5} 92 {x} 108 Q{x+5} 93 {x+10} 109" fill="none" stroke="{AMBER}" stroke-width="1.4"/>'
    s+=text(x,133,'Temperature + cutoff',11.2)+text(x,148,'Known calibration only',11.4)
    x=208;s+=circle(x,103,16,BLUE,BLUE+'0D')+f'<path d="M{x-8} 104 L{x-2} 110 L{x+9} 96" fill="none" stroke="{BLUE}" stroke-width="1.8"/>'
    s+=text(x,133,'Fixed-rule evaluation',11.2)+text(x,148,'Untouched held donor',11.4)
    s+=line(84,103,181,103,'#8296A3',1.2)+f'<path d="M176 99 L182 103 L176 107" fill="none" stroke="#8296A3" stroke-width="1.2"/>'
    return wrap(266,153,s)
def scores():
    s=''
    for y,c,title,desc in [(19,TEAL,'Maximum probability','1 − largest class probability'),(71,AMBER,'Normalized entropy','Class uncertainty / log(K)'),(123,BLUE,'Nearest centroid','Distance to training mean')]:
        if y==19:
            for i,h in enumerate([7,16,28,11]):s+=f'<path d="M{5+i*7} {y+12} v{-h}" stroke="{c}" stroke-width="4"/>'
        elif y==71:
            for i,h in enumerate([16,19,17,18]):s+=f'<path d="M{5+i*7} {y+12} v{-h}" stroke="{c}" stroke-width="4"/>'
        else:
            for dx,dy in [(0,0),(5,-8),(10,5)]:s+=circle(12+dx,y+dy,1.7,c,c,.5)
            s+=circle(15,y,4,c)+line(19,y,31,y,ROSE)+star(35,y,ROSE)
        s+=text(47,y-3,title,11.8,'start')+text(47,y+13,desc,10.8,'start')
    return wrap(184,147,s)
BUILDERS={'A':donor,'C':fitting,'E':scores}
def write_sources(root):
    out=root/'figures/schematics';out.mkdir(exist_ok=True)
    for key,fn in BUILDERS.items():(out/f'Figure1{key}.svg').write_text(fn()+'\n')
def integrate(svg,root,rectangles,size,only=None):
    tree=ET.fromstring(svg);width,height=[v*72 for v in size]
    for key in BUILDERS:
        if only is not None and key!=only:continue
        x,y,w,h=rectangles[ord(key)-65];asset=ET.fromstring((root/'figures/schematics'/f'Figure1{key}.svg').read_text())
        aw,ah=[float(v) for v in asset.attrib['viewBox'].split()[2:]]
        scale=min(w*width/aw,h*height/ah)
        xx=x*width+(w*width-aw*scale)/2;yy=(1-y-h)*height+(h*height-ah*scale)/2
        g=ET.SubElement(tree,f'{{{NS}}}g',{'id':f'original-svg-schematic-{key}','transform':f'translate({xx:.8f},{yy:.8f}) scale({scale:.8f})'})
        for child in asset:g.append(copy.deepcopy(child))
    return ET.tostring(tree,encoding='unicode')
def render(svg,pdf=None,png=None,crop=None):
    helper=Path(__file__).with_name('svg_render.py')
    args=[os.environ.get('SC_VECTOR_PYTHON',shutil.which('python3')),str(helper),str(svg)]
    if pdf:args+=['--pdf',str(pdf)]
    if png:args+=['--png',str(png)]
    if crop:args+=['--crop',*[str(v) for v in crop]]
    subprocess.run(args,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
