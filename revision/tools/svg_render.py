"""Deterministic SVG-to-PDF production conversion using existing Inkscape."""
from pathlib import Path
import argparse,subprocess,tempfile,os
import pikepdf
from xml.etree import ElementTree as ET
p=argparse.ArgumentParser();p.add_argument('svg',type=Path);p.add_argument('--pdf',type=Path);p.add_argument('--png',type=Path);p.add_argument('--crop',type=float,nargs=4);a=p.parse_args()
env=os.environ.copy();env['SOURCE_DATE_EPOCH']='1792108800'
with tempfile.TemporaryDirectory(prefix='array-svg-') as tmp:
 src=a.svg
 if a.crop:
  tree=ET.fromstring(src.read_text());x,y,w,h=a.crop;tree.set('viewBox',f'{x:.8f} {y:.8f} {w:.8f} {h:.8f}');tree.set('width',f'{w:.8f}pt');tree.set('height',f'{h:.8f}pt')
  src=Path(tmp)/'crop.svg';ET.ElementTree(tree).write(src,encoding='unicode')
 if a.pdf:
  out=Path(tmp)/'render.pdf';subprocess.run(['inkscape',str(src),'--export-type=pdf','--export-filename='+str(out)],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True)
  with pikepdf.open(out) as pdf:
   for k in ['/CreationDate','/ModDate']:
    if k in pdf.docinfo:del pdf.docinfo[k]
   if '/Metadata' in pdf.Root:del pdf.Root['/Metadata']
   pdf.save(a.pdf,deterministic_id=True)
 if a.png:subprocess.run(['inkscape',str(src),'--export-type=png','--export-dpi=300','--export-filename='+str(a.png)],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True)
