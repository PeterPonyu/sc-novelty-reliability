"""Recount saved predictions with independent rank and threshold calculations."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import anndata as ad
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]

def probabilities(z,t):
    zz=z/t;zz=zz-zz.max(axis=1,keepdims=True);e=np.exp(zz);return e/e.sum(axis=1,keepdims=True)

def auc_by_rank(y,s):
    n1=int(y.sum());n0=len(y)-n1
    return float((rankdata(s)[y].sum()-n1*(n1+1)/2)/(n1*n0))

def ap_grouped_ties(y,s):
    order=np.argsort(-s,kind='stable');score=s[order];truth=y[order]
    endpoints=np.r_[np.flatnonzero(score[:-1]!=score[1:]),len(score)-1]
    tp=np.cumsum(truth)[endpoints];prec=tp/(endpoints+1);rec=tp/truth.sum()
    return float(np.sum(np.diff(np.r_[0,rec])*prec))

def assert_close(a,b,tol=2e-12):
    if a is None or b is None:
        if a is not b:raise ValueError('undefined value mismatch')
        return 0.
    diff=abs(float(a)-float(b))
    if diff>tol:raise ValueError(f'metric mismatch {a} vs {b}')
    return diff

def score_values(z,d,t):
    p=probabilities(z,1);pt=probabilities(z,t)
    entropy=lambda a:-(a*np.log(np.maximum(a,1e-300))).sum(1)/np.log(a.shape[1])
    return {'msp_raw':1-p.max(1),'msp_temperature':1-pt.max(1),'entropy_raw':entropy(p),'entropy_temperature':entropy(pt),'centroid_distance':d}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scores-only',action='store_true',help='Recount predictions without claiming raw-metadata verification');parser.add_argument('--pulmonary-input',type=Path)
    args=parser.parse_args()
    plan=json.loads((ROOT/'experiments/analysis-plan.json').read_text())
    metrics=ops=mixtures=control_metrics=calibration_checks=0;maximum=0.;atlases={}
    for folder in sorted((ROOT/'experiments').glob('SC*')):
        cfg=plan['atlases'][folder.name]
        if not (folder/'rows').exists():continue
        raw=None
        if not args.scores_only:
            inp=args.pulmonary_input if folder.name=='SC31' and args.pulmonary_input else Path(cfg['atlas_path'])
            with inp.open('rb') as f:input_hash=hashlib.file_digest(f,'sha256').hexdigest()
            if input_hash!=cfg['atlas_sha256']:raise ValueError('Raw input hash mismatch')
            raw=ad.read_h5ad(inp,backed='r');labels=raw.obs[cfg['cell_type_column']].astype(str).to_numpy();donors=raw.obs[cfg['donor_column']].astype(str).to_numpy()
        ids=set();scored=0
        for p in sorted((folder/'rows').glob('*.json')):
            r=json.loads(p.read_text());key=(r['type'],r['test_donor'],r['calibration_donor'])
            if key in ids:raise ValueError('duplicate task')
            ids.add(key)
            if r['status']!='SCORED':continue
            scored+=1;z=np.load(folder/'scores'/f"{r['row_id']}.npz")
            tr,ca,te=z['training_row'],z['calibration_row'],z['test_row']
            if np.intersect1d(tr,ca).size or np.intersect1d(tr,te).size or np.intersect1d(ca,te).size:raise ValueError('row leakage')
            classes=np.array(r['training_classes']);ky=z['test_y']>=0
            if raw is not None:
                if set(donors[tr])!=set(r['training_donors']) or set(donors[ca])!={r['calibration_donor']} or set(donors[te])!={r['test_donor']}:raise ValueError('role mismatch')
                if np.any(labels[tr]==r['type']) or np.any(labels[ca]==r['type']):raise ValueError('target enters fitted partition')
                if not np.array_equal(labels[te]==r['type'],z['test_y']==-1):raise ValueError('novel labels mismatch')
                if not np.array_equal(labels[te][ky],classes[z['test_y'][ky]]):raise ValueError('known test labels mismatch')
                if not np.array_equal(labels[ca],classes[z['calibration_y']]):raise ValueError('calibration labels mismatch')
            cs=score_values(z['calibration_logits'],z['calibration_distance'],r['temperature']);ts=score_values(z['test_logits'],z['test_distance'],r['temperature'])
            for suffix,temp in [('raw',1),('temperature',r['temperature'])]:
                pp=probabilities(z['test_logits'][ky],temp);yy=z['test_y'][ky]
                truth=np.zeros_like(pp);truth[np.arange(len(yy)),yy]=1
                correct_known=pp.argmax(1)==yy;confidence=pp.max(1)
                assignments=np.minimum(np.floor(confidence*10).astype(int),9)
                ece=0.
                for b in range(10):
                    m=assignments==b
                    if m.any():ece+=abs(float((correct_known[m]-confidence[m]).sum()))/len(yy)
                vals={'nll':float(-np.log(np.maximum(pp[np.arange(len(yy)),yy],1e-12)).mean()),
                      'brier':float(np.square(pp-truth).sum(1).mean()),'ece':ece,'accuracy':float(correct_known.mean())}
                for key,value in vals.items():
                    maximum=max(maximum,assert_close(value,r['known_test_calibration_'+suffix][key]));calibration_checks+=1
            y=z['test_y']==-1;correct=(z['test_logits'].argmax(1)==z['test_y'])&(~y)
            for name,s in ts.items():
                d=r['scores'][name]
                maximum=max(maximum,assert_close(auc_by_rank(y,s),d['auroc']),assert_close(ap_grouped_ties(y,s),d['average_precision']));metrics+=1
                for op in d['operating_points']:
                    sorted_cal=np.sort(cs[name]);q=op['nominal_known_coverage'];threshold=float(sorted_cal[int(np.ceil(q*(len(sorted_cal)-1)))])
                    maximum=max(maximum,assert_close(threshold,op['threshold']))
                    keep=s<=threshold;ak=float(keep[~y].mean());an=float(keep[y].mean());ek=float(np.mean(keep[~y]&(~correct[~y])))
                    for key,value in [('held_known_coverage',ak),('novel_acceptance',an),('held_mixture_coverage',keep.mean()),('selective_risk',1-correct[keep].mean() if keep.any() else None)]:
                        maximum=max(maximum,assert_close(value,op[key]))
                    for row in op['prevalence_sensitivity']:
                        pi=row['novel_prevalence'];den=(1-pi)*ak+pi*an;rr=((1-pi)*ek+pi*an)/den if den else None
                        maximum=max(maximum,assert_close(den,row['coverage']),assert_close(rr,row['risk']));mixtures+=1
                    ops+=1
            if 'permuted_training_labels' in r:
                ctrl=r['permuted_training_labels'];zz=np.load(folder/'scores'/f"{r['row_id']}_label_control.npz")
                vals=score_values(zz['test_logits'],zz['test_distance'],ctrl['temperature'])
                for name,d in ctrl['scores'].items():
                    maximum=max(maximum,assert_close(auc_by_rank(zz['test_y']==-1,vals[name]),d['auroc']));control_metrics+=1
        under=json.loads((folder/'underpowered.json').read_text());inventory=json.loads((folder/'input_inventory.json').read_text())
        if len(ids)+len(under)!=inventory['planned_rows']:raise ValueError('incomplete universe')
        if raw is not None:
            for row in under:
                n_total=int(np.sum(labels==row['type']));held=int(np.sum((labels==row['type'])&(donors==row['test_donor'])))
                if n_total>=60 and held>=30:raise ValueError('unjustified count exclusion')
            raw.file.close()
        atlases[folder.name]={'scored_tasks':scored,'explicit_count_exclusions':len(under),'planned_rows':inventory['planned_rows'],'row_disjointness':'PASS','raw_metadata_role_and_label_checks':'NOT_RECHECKED_SCORES_ONLY' if args.scores_only else 'PASS'}
    result={'status':'PASS','atlases':atlases,'independently_recounted_score_metrics':metrics,'threshold_operating_points':ops,
            'prevalence_scenarios':mixtures,'training_label_control_metrics':control_metrics,'known_calibration_metrics':calibration_checks,'maximum_absolute_difference':maximum,
            'scope':'Saved-score arithmetic; raw-metadata role/target checks '+('not rechecked in explicit scores-only mode.' if args.scores_only else 'verified against hash-matched input.')+' Separate cold-fit check covers selected original fits; this is not an independent biological replication.'}
    (ROOT/'evidence/prediction-recount.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
