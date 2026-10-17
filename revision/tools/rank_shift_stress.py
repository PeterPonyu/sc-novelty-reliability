"""Controlled score-transport stress; no new cells or biological replication."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.special import expit,logit,softmax
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'experiments/rank-shift-plan.json'

def run():
    plan=json.loads(PLAN.read_text());rows=[]
    for p in sorted((ROOT/'experiments').glob('SC*/rows/*.json')):
        d=json.loads(p.read_text())
        if d['status']!='SCORED':continue
        z=np.load(p.parent.parent/'scores'/f"{d['row_id']}.npz")
        cal=1-softmax(z['calibration_logits']/d['temperature'],axis=1).max(1)
        test=1-softmax(z['test_logits']/d['temperature'],axis=1).max(1)
        target=z['test_y']==-1;known=~target;correct=(z['test_logits'].argmax(1)==z['test_y'])&known
        if np.any(test<0) or np.any(test>1) or np.any(cal<0) or np.any(cal>1):
            raise ValueError('score outside the unit interval')
        threshold=float(np.quantile(cal,.8,method='higher'));anchor_auc=roc_auc_score(target,test)
        for delta in plan['log_odds_shifts']:
            factor=np.exp(delta)
            transformed=factor*test/(1-test+factor*test);keep=transformed<=threshold
            # Positive control: transforming calibration too preserves every
            # acceptance decision, because the transform is strictly increasing.
            transported_threshold=float(factor*threshold/(1-threshold+factor*threshold))
            consistent=transformed<=transported_threshold
            if not np.array_equal(consistent,test<=threshold):raise ValueError('joint-transform acceptance invariance failed')
            rows.append({'atlas':d['atlas'],'row_id':d['row_id'],'type':d['type'],
                         'test_donor':d['test_donor'],'calibration_donor':d['calibration_donor'],
                         'shift':delta,'auroc':float(roc_auc_score(target,transformed)),
                         'auroc_difference':float(roc_auc_score(target,transformed)-anchor_auc),
                         'known_coverage':float(keep[known].mean()),'novel_acceptance':float(keep[target].mean()),
                         'mixture_coverage':float(keep.mean()),'risk':float(1-correct[keep].mean()) if keep.any() else None,
                         'accepted_n':int(keep.sum()),'joint_transform_decisions_identical':True})
    df=pd.DataFrame(rows);df.to_csv(ROOT/'evidence/rank-shift-stress.csv',index=False)
    summary={'rows':len(df),'prediction_tasks':df.row_id.nunique(),'shifts':plan['log_odds_shifts'],
             'maximum_absolute_auroc_difference':float(df.auroc_difference.abs().max()),
             'joint_transform_decisions_identical':bool(df.joint_transform_decisions_identical.all()),
             'zero_acceptance_rows':int(sum(df.accepted_n==0)),
             'plan_sha256':hashlib.sha256(PLAN.read_bytes()).hexdigest(),
             'interpretation':'Controlled monotone score shift of held scores only; illustrates rank versus threshold transport. No new cohort or biological perturbation.'}
    (ROOT/'evidence/rank-shift-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':run()
