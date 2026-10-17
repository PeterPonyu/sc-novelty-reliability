"""Relocate the unchanged scientific runner and exact input into a fresh output folder.

Only the input location is changed in a copied plan, with parent plan identity
recorded separately. The original plan and saved outputs remain untouched.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit-fit-groups',type=int)
    a=p.parse_args();inp=a.input.resolve();dest=a.output.resolve()
    if dest.exists():raise SystemExit('Output must be a new directory; existing records are never overwritten.')
    planpath=ROOT/'experiments/analysis-plan.json';plan=json.loads(planpath.read_text());cfg=plan['atlases']['SC31']
    with inp.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=cfg['atlas_sha256']:raise SystemExit('Exact original input hash required.')
    (dest/'experiments').mkdir(parents=True);(dest/'tools').mkdir();(dest/'qa').mkdir()
    cfg['atlas_path']=str(inp);(dest/'experiments/analysis-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    shutil.copy2(ROOT/'tools/run_prediction_extension.py',dest/'tools/run_prediction_extension.py')
    (dest/'experiments/relocation-receipt.json').write_text(json.dumps({'parent_plan_sha256':hashlib.sha256(planpath.read_bytes()).hexdigest(),'only_plan_field_changed':'atlases.SC31.atlas_path','input_sha256':actual,'scientific_settings_changed':False},indent=2)+'\n')
    cmd=[sys.executable,str(dest/'tools/run_prediction_extension.py'),'--atlas','SC31']
    if a.limit_fit_groups is not None:cmd+=['--limit-fit-groups',str(a.limit_fit_groups)]
    env=os.environ.copy()
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:env[key]='2'
    subprocess.run(cmd,cwd=dest,env=env,check=True)
if __name__=='__main__':main()
