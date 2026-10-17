"""Fresh donor-isolated predictions and calibration-selected rejection thresholds.

Run on an explicitly specified primary atlas; no original result is overwritten.
The analysis plan is saved before fitting. A train set/target is fit once and may
be reused for the two opposite calibration/test assignments with identical train
donors. Every observed type and ordered donor pair receives an explicit status.
"""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, os, platform, time, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['NUMBA_DISABLE_CUDA'] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(ROOT/'qa/numba-cache')
os.environ['MPLCONFIGDIR'] = str(ROOT/'qa/matplotlib-cache')
import anndata as ad
import numpy as np
import scanpy as sc
import scipy
import sklearn
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

SCORES = ['msp_raw','msp_temperature','entropy_raw','entropy_temperature','centroid_distance']

def sha(path):
    with Path(path).open('rb') as h: return hashlib.file_digest(h,'sha256').hexdigest()

def dump(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def dense(x): return np.asarray(x.toarray() if hasattr(x,'toarray') else x,dtype=np.float64)

def logits(model,x):
    z=model.decision_function(x)
    # The evaluated atlases have >2 classes; this branch is explicit for reuse.
    return np.column_stack([np.zeros_like(z),z]) if z.ndim==1 else z

def calibration(logit,y,temp):
    p=softmax(logit/temp,axis=1); conf=p.max(1); correct=p.argmax(1)==y
    bins=[]
    for j in range(10):
        mask=(conf>=j/10)&((conf<(j+1)/10) if j<9 else (conf<=1))
        bins.append({'bin':j,'n':int(mask.sum()),'confidence':float(conf[mask].mean()) if mask.any() else None,'accuracy':float(correct[mask].mean()) if mask.any() else None})
    return {'nll':float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),
            'brier':float((np.sum(p*p,axis=1)+1-2*p[np.arange(len(y)),y]).mean()),
            'ece':float(sum(b['n']/len(y)*abs(b['confidence']-b['accuracy']) for b in bins if b['n'])),
            'accuracy':float(correct.mean()),'bins':bins}

def get_scores(logit,distance,t):
    p0=softmax(logit,axis=1); pt=softmax(logit/t,axis=1)
    ent=lambda p: -np.sum(p*np.log(np.maximum(p,1e-300)),axis=1)/np.log(p.shape[1])
    return {'msp_raw':1-p0.max(1),'msp_temperature':1-pt.max(1),
            'entropy_raw':ent(p0),'entropy_temperature':ent(pt),'centroid_distance':distance}

def evaluate_arrays(cal_logit,cal_y,test_logit,test_y,cal_distance,test_distance,plan):
    objective=lambda t: calibration_nll(cal_logit,cal_y,t)
    fit=minimize_scalar(objective,bounds=plan['temperature_bounds'],method='bounded')
    if not fit.success: raise RuntimeError('temperature fit failed')
    temp=float(fit.x)
    known=test_y>=0; novel=test_y==-1
    if not np.all(known|novel): raise ValueError('evaluation contains undeclared labels')
    correct=(test_logit.argmax(1)==test_y)&known
    cs=get_scores(cal_logit,cal_distance,temp); ts=get_scores(test_logit,test_distance,temp)
    result={'temperature':temp,'known_test_calibration_raw':calibration(test_logit[known],test_y[known],1),
            'known_test_calibration_temperature':calibration(test_logit[known],test_y[known],temp),
            'calibration_fit_raw':calibration(cal_logit,cal_y,1),
            'calibration_fit_temperature':calibration(cal_logit,cal_y,temp),'scores':{}}
    for name in SCORES:
        score=ts[name]; record={'auroc':float(roc_auc_score(novel,score)),
                                'average_precision':float(average_precision_score(novel,score)),
                                'novel_prevalence':float(novel.mean()),'operating_points':[]}
        for q in plan['nominal_known_coverage']:
            threshold=float(np.quantile(cs[name],q,method='higher'))
            keep=score<=threshold
            ak=float(keep[known].mean()); an=float(keep[novel].mean())
            # Error numerator per known cell, before conditioning on acceptance.
            ek=float(np.mean(keep[known]&(~correct[known])))
            k=max(1,int(len(score)*q)); top=np.argsort(score,kind='stable')[:k]
            op={'nominal_known_coverage':q,'threshold':threshold,
                'calibration_realized_coverage':float(np.mean(cs[name]<=threshold)),
                'held_known_coverage':ak,'novel_acceptance':an,'held_mixture_coverage':float(keep.mean()),
                'selective_risk':float(1-correct[keep].mean()) if keep.any() else None,
                'accepted_n':int(keep.sum()),'accepted_known_error_per_known':ek,
                'topk_coverage':float(k/len(score)),'topk_risk':float(1-correct[top].mean()),
                'forced_risk':float(1-correct.mean()),'prevalence_sensitivity':[]}
            for pi in plan['mixture_prevalence']:
                den=(1-pi)*ak+pi*an
                op['prevalence_sensitivity'].append({'novel_prevalence':pi,'coverage':den,
                    'risk':((1-pi)*ek+pi*an)/den if den>0 else None})
            record['operating_points'].append(op)
        result['scores'][name]=record
    return result

def calibration_nll(z,y,t):
    p=softmax(z/t,axis=1)
    return float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())

def run(atlas,limit=None):
    planfile=ROOT/'experiments/analysis-plan.json';plan=json.loads(planfile.read_text())
    cfg=plan['atlases'][atlas]; inp=Path(cfg['atlas_path']); out=ROOT/'experiments'/atlas
    out.mkdir(exist_ok=True)
    if not inp.is_file():
        dump(out/'availability.json',{'status':'INPUT_UNAVAILABLE','expected_path':str(inp),'expected_sha256':cfg['atlas_sha256']});return
    observed_hash=sha(inp)
    if observed_hash!=cfg['atlas_sha256']: raise ValueError('input hash mismatch')
    started=time.monotonic(); a=ad.read_h5ad(inp)
    donor_col,type_col=cfg['donor_column'],cfg['cell_type_column']
    donor_universe=sorted(set(cfg['donor_cycle']))
    original_rows=np.flatnonzero(np.isin(a.obs[donor_col].astype(str),donor_universe))
    a=a[original_rows].copy() if len(original_rows)!=a.n_obs else a
    preview=dense(a.X[:80]); counts=bool(np.allclose(preview,np.round(preview)) and preview.max()>20)
    if counts: sc.pp.normalize_total(a,target_sum=1e4);sc.pp.log1p(a)
    y=a.obs[type_col].astype(str).to_numpy(); donors=a.obs[donor_col].astype(str).to_numpy()
    types=sorted(set(y)); jobs=[];statuses=[]
    for test,cal in itertools.permutations(donor_universe,2):
        train=tuple(d for d in donor_universe if d not in (test,cal))
        for target in types:
            row={'atlas':atlas,'type':target,'test_donor':test,'calibration_donor':cal,'training_donors':list(train),
                 'n_total':int(np.sum(y==target)),'n_novel':int(np.sum((y==target)&(donors==test)))}
            key=hashlib.sha256(json.dumps([atlas,target,test,cal]).encode()).hexdigest()[:16]
            row['row_id']=key
            if row['n_total']<60 or row['n_novel']<30:
                row.update(status='UNDERPOWERED',reason='target support below 60 total or 30 held donor')
                statuses.append(row)
            else: jobs.append(row)
    groups={}
    for row in jobs: groups.setdefault((row['type'],tuple(row['training_donors'])),[]).append(row)
    # Same ordering is retained on resume; each cached row is bound to this plan.
    dump(out/'input_inventory.json',{'input_sha256':observed_hash,'analysis_plan_sha256':sha(planfile),
         'n_cells':a.n_obs,'n_genes':a.n_vars,'n_types':len(types),'donor_universe':donor_universe,
         'counts_transform_applied':counts,'n_ordered_pairs':len(donor_universe)*(len(donor_universe)-1),
         'planned_rows':len(jobs)+len(statuses),'eligible_by_target_counts':len(jobs),'unique_fit_groups':len(groups),
         'donor_type_counts':{t:{d:int(np.sum((y==t)&(donors==d))) for d in donor_universe} for t in types},
         'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'scanpy':sc.__version__,'sklearn':sklearn.__version__},
         'platform':platform.platform()})
    dump(out/'underpowered.json',statuses)
    n_new=0
    for (target,train_donors),group in groups.items():
        if limit is not None and n_new>=limit: break
        if all((out/'rows'/f"{r['row_id']}.json").exists() for r in group):continue
        t0=time.monotonic();train_mask=np.isin(donors,train_donors)&(y!=target)
        classes=sorted(set(y[train_mask]));class_index={c:i for i,c in enumerate(classes)}
        if len(classes)<2: raise ValueError('fewer than two training classes')
        train=a[train_mask].copy();sc.pp.highly_variable_genes(train,n_top_genes=min(plan['hvg_n'],a.n_vars-1))
        hvg=np.flatnonzero(train.var['highly_variable'].to_numpy())
        xt=dense(train[:,hvg].X);del train
        scale=StandardScaler().fit(xt);xt=scale.transform(xt)
        pca=PCA(n_components=min(plan['pca_n'],min(xt.shape)-1),random_state=plan['representation_seed'],svd_solver=plan['pca_solver'])
        ztrain=pca.fit_transform(xt);del xt
        ytrain=np.array([class_index[v] for v in y[train_mask]])
        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter('always',ConvergenceWarning)
            model=LogisticRegression(max_iter=plan['logistic_max_iter'],random_state=plan['representation_seed']).fit(ztrain,ytrain)
        converged=not any(issubclass(w.category,ConvergenceWarning) for w in warns)
        centers=np.stack([ztrain[ytrain==i].mean(0) for i in range(len(classes))])
        cached={}
        for d in set(donor_universe)-set(train_donors):
            ids=np.flatnonzero(donors==d);z=pca.transform(scale.transform(dense(a[ids,hvg].X)))
            cy=np.array([class_index.get(v,-1 if v==target else -2) for v in y[ids]])
            cached[d]=(ids,z,cy,logits(model,z),cdist(z,centers).min(1))
        shuffled_model=None
        for row in group:
            cids,cz,cy,cl,cd=cached[row['calibration_donor']]
            tids,tz,ty,tl,td=cached[row['test_donor']]
            cm=cy>=0;tm=ty!=-2;tk=ty>=0
            common={**row,'analysis_plan_sha256':sha(planfile),'input_sha256':observed_hash,
                    'n_train':int(train_mask.sum()),'n_calibration':int(cm.sum()),'n_known':int(tk.sum()),
                    'excluded_calibration_target':int(np.sum(cy==-1)),
                    'excluded_calibration_other':int(np.sum(cy==-2)),'excluded_test_other':int(np.sum(ty==-2)),
                    'training_classes':classes,'n_hvg':len(hvg),'pca_solver_resolved':pca._fit_svd_solver,
                    'readout_n_iter':int(model.n_iter_.max()),'readout_converged':converged,
                    'fit_wall_seconds':time.monotonic()-t0}
            if cm.sum()<2 or tk.sum()<2:
                common.update(status='UNDERPOWERED',reason='insufficient reference-present known cells')
                dump(out/'rows'/f"{row['row_id']}.json",common);continue
            detail=evaluate_arrays(cl[cm],cy[cm],tl[tm],ty[tm],cd[cm],td[tm],plan)
            common.update(status='SCORED' if converged else 'NONCONVERGED',**detail)
            scorespath=out/'scores'/f"{row['row_id']}.npz";scorespath.parent.mkdir(exist_ok=True)
            np.savez_compressed(scorespath,calibration_logits=cl[cm],calibration_y=cy[cm],test_logits=tl[tm],test_y=ty[tm],
                calibration_distance=cd[cm],test_distance=td[tm],calibration_row=original_rows[cids[cm]],test_row=original_rows[tids[tm]],
                training_row=original_rows[np.flatnonzero(train_mask)],hvg_indices=hvg)
            common['scores_sha256']=sha(scorespath)
            anchor=cfg['primary_anchor']
            if row['test_donor']==anchor['test'] and row['calibration_donor']==anchor['calibration']:
                perm_seed=int(hashlib.sha256(json.dumps([plan['seed'],atlas,target,train_donors]).encode()).hexdigest()[:8],16)
                rng=np.random.default_rng(perm_seed);yp=ytrain.copy();train_groups=donors[train_mask]
                for d in train_donors:
                    ix=np.flatnonzero(train_groups==d);yp[ix]=rng.permutation(yp[ix])
                with warnings.catch_warnings(record=True) as sw:
                    warnings.simplefilter('always',ConvergenceWarning)
                    shuffled_model=LogisticRegression(max_iter=plan['logistic_max_iter'],random_state=plan['representation_seed']).fit(ztrain,yp)
                slc=logits(shuffled_model,cz[cm]);slt=logits(shuffled_model,tz[tm])
                control=evaluate_arrays(slc,cy[cm],slt,ty[tm],cd[cm],td[tm],plan)
                control['seed']=perm_seed;control['readout_converged']=not any(issubclass(w.category,ConvergenceWarning) for w in sw)
                # Distance is unchanged by this label permutation because the
                # control tests the readout only. Do not count it as a null detector.
                control['scores'].pop('centroid_distance')
                common['permuted_training_labels']=control
                cp=out/'scores'/f"{row['row_id']}_label_control.npz"
                np.savez_compressed(cp,calibration_logits=slc,calibration_y=cy[cm],test_logits=slt,test_y=ty[tm],
                    calibration_distance=cd[cm],test_distance=td[tm])
                common['control_scores_sha256']=sha(cp)
            dump(out/'rows'/f"{row['row_id']}.json",common)
        n_new+=1
        print(json.dumps({'atlas':atlas,'target':target,'train':train_donors,'rows':len(group),'new_fit_group':n_new,
                          'converged':converged,'seconds':round(time.monotonic()-t0,2)}),flush=True)
    written=[json.loads(p.read_text()) for p in sorted((out/'rows').glob('*.json'))] if (out/'rows').exists() else []
    summary={'status':'COMPLETE' if len(written)==len(jobs) else 'IN_PROGRESS','planned_rows':len(jobs)+len(statuses),
             'completed_rows':len(written)+len(statuses),'scored_rows':sum(r['status']=='SCORED' for r in written),
             'nonconverged_rows':sum(r['status']=='NONCONVERGED' for r in written),'total_wall_seconds_this_call':time.monotonic()-started,
             'analysis_plan_sha256':sha(planfile),'runner_sha256':sha(__file__)}
    dump(out/'run-status.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--atlas',required=True);ap.add_argument('--limit-fit-groups',type=int)
    args=ap.parse_args();run(args.atlas,args.limit_fit_groups)
