"""Normalize frozen evidence and extend reporting-unit sensitivity transparently."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
RESULTS=(ROOT/'historical-results') if (ROOT/'historical-results').is_dir() else ROOT.parents[3]/'research/results'
EVIDENCE=ROOT/'evidence'
ATLAS={'SC31':'Pulmonary','SC32':'Hepatic','SC33_adult':'Adult cardiac'}
ORDER=list(ATLAS.values())
SOURCES={}

def read(path):
    path=Path(path);SOURCES[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())

def write_json(name,d): (EVIDENCE/name).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def rotation_rows(d,k,atlas):
    out=[]
    for rr in d['rotations']:
        for r in rr['types']:
            row={'atlas':ATLAS[atlas],'scheme':k,'type':r['type'],**rr['rotation'],'status':r['status'],
                 'n_total':r['n_total'],'n_held':r['n_in_held_donor']}
            if r['status']=='scored':
                row.update(auroc=r['auroc'],ci_low=r['auroc_ci'][0],ci_high=r['auroc_ci'][1],temperature=r['temperature'])
                for q,x in r['coverage_risk'].items():row['risk_'+q]=x['risk']
            else: row['reason']=r['reason']
            row['training_donors']='|'.join(row['training_donors']);out.append(row)
    return out

def normalise():
    a1=read(RESULTS/'SC31_SC33/sc31_33_donor_rotation_robustness.json')
    rows=[]
    for name,a in a1['atlases'].items(): rows+=rotation_rows(a,1,name)
    for p in sorted((RESULTS/'SC31_SC33/e2_multi_split').glob('e2_SC*.json')):
        d=read(p);rows+=rotation_rows(d,int(d['calibration_offset_k']),d['atlas'])
    pairs=pd.DataFrame(rows)
    if pairs.duplicated(['atlas','type','test_donor','calibration_donor']).any():raise ValueError('duplicate role assignment')
    pairs.to_csv(EVIDENCE/'historical-all-donor-pairs.csv',index=False)
    bounded=read(RESULTS/'SC31_SC33/sc31_33_bounded_sensitivity.json')
    rows=[]
    for r in bounded['rows']:
        row={k:r[k] for k in ['atlas','type','test_donor','calibration_donor','status','specification_id']}
        row['atlas']=ATLAS[row['atlas']]
        if r['status']=='scored':row.update(auroc=r['auroc'],ci_low=r['auroc_ci'][0],ci_high=r['auroc_ci'][1],temperature=r['temperature'])
        rows.append(row)
    pd.DataFrame(rows).to_csv(EVIDENCE/'historical-specifications.csv',index=False)
    rows=[]
    for suffix in ['geneformer','geneformer316','scgpt','scfoundation','uce','cellplm']:
        d=read(RESULTS/f'A_PLUS/a_plus_fm_compatibility_sc31_{suffix}.json')
        for r in d['per_type']:
            row={'model':d['model'],'type':r['type'],'status':r['status']}
            for k in ['auroc','brier','ece','temperature','n_in_held_donor','n_known','readout_n_iter_max']:row[k]=r[k]
            for op in r['risk_coverage']:row[f"risk_{op['requested_coverage']:.2f}"]=op['risk']
            rows.append(row)
    pd.DataFrame(rows).to_csv(EVIDENCE/'historical-frozen-representations.csv',index=False)
    d=read(RESULTS/'A_PLUS/a_plus_portability_gate_3family_2026-08-29.json')
    rows=[]
    for a in d['datasets']:
        for m in a['models']:
            for t in m['targets']:
                row={'dataset':a['dataset'],'model':m['model'],'type':t['target'],'status':t['status'],
                     'held_batch':a['split']['held_batch'],'calibration_batch':a['split']['calibration_batch']}
                for k in ['auroc','brier','ece','temperature','n_target_in_held','n_known_in_held']:row[k]=t[k]
                for op in t['risk_coverage']:row[f"risk_{op['requested_coverage']:.2f}"]=op['risk']
                rows.append(row)
    pd.DataFrame(rows).to_csv(EVIDENCE/'historical-portability.csv',index=False)
    scored=pairs[pairs.status=='scored'];summary={}
    for a,df in scored.groupby('atlas',sort=False):
        donor=df.groupby('test_donor').auroc.median()
        type_med=df.groupby('type').auroc.median()
        spread=df.groupby('type').auroc.agg(['min','max'])
        summary[a]={'planned_rows':int(sum(pairs.atlas==a)),'scored_rows':len(df),
                    'ordered_pairs':int(df[['test_donor','calibration_donor']].drop_duplicates().shape[0]),
                    'record_median':float(df.auroc.median()),'equal_held_donor_median':float(donor.median()),
                    'equal_held_donor_mean':float(donor.mean()),'donor_medians':donor.to_dict(),
                    'donor_median_range':[float(donor.min()),float(donor.max())],
                    'type_median_range':[float(type_med.min()),float(type_med.max())],
                    'types_crossing_chance':spread[(spread['min']<.5)&(spread['max']>.5)].index.tolist()}
    write_json('historical-summary.json',summary)

def corr(x,y):
    x=x-x.mean(-1,keepdims=True);y=y-y.mean(-1,keepdims=True)
    return (x*y).sum(-1)/np.sqrt((x*x).sum(-1)*(y*y).sum(-1))

def aggregation():
    p=EVIDENCE/'deep_type_level.csv';df=pd.read_csv(p);y=df.auroc.to_numpy();g=df.atlas.to_numpy()
    rng=np.random.default_rng(20261016);ix=np.empty((20000,len(df)),dtype=int)
    for a in ORDER:
        ids=np.flatnonzero(g==a);ix[:,ids]=rng.choice(ids,(20000,len(ids)),replace=True)
    def center(x,groups=g):
        out=x.copy()
        for a in set(groups):
            mask=groups==a;out[...,mask]-=out[...,mask].mean(-1,keepdims=True)
        return out
    results={};omit=[]
    for name,x in [('log10_distance',np.log10(df.nearest_centroid_euclidean.to_numpy())),
                   ('raw_distance',df.nearest_centroid_euclidean.to_numpy()),
                   ('silhouette',df.mean_silhouette_type_vs_rest.to_numpy())]:
        draws_pool=corr(x[ix],y[ix]);draws_center=corr(center(x[ix]),center(y[ix]));delta=draws_pool-draws_center
        within={}
        for a in ORDER:
            m=g==a;d=corr(x[ix][:,m],y[ix][:,m]);valid=np.isfinite(d)
            within[a]={'r':float(corr(x[m],y[m])),'interval':np.quantile(d[valid],[.025,.975]).tolist(),'finite':int(valid.sum())}
        ctotal=float(np.sum((x-x.mean())*(y-y.mean())));cw=float(np.sum(center(x)*center(y)))
        cb=float(sum(np.sum(g==a)*(x[g==a].mean()-x.mean())*(y[g==a].mean()-y.mean()) for a in ORDER))
        if not np.isclose(ctotal,cw+cb,atol=1e-12):raise ValueError('cross-product identity failed')
        results[name]={'pooled_r':float(corr(x,y)),'centered_r':float(corr(center(x),center(y))),
          'paired_difference':float(corr(x,y)-corr(center(x),center(y))),
          'paired_difference_interval':np.quantile(delta,[.025,.975]).tolist(),
          'pooled_interval':np.quantile(draws_pool,[.025,.975]).tolist(),
          'centered_interval':np.quantile(draws_center,[.025,.975]).tolist(),
          'within':within,'cross_product':{'total':ctotal,'within':cw,'between':cb}}
        for a in ORDER:
            m=g!=a;omit.append({'covariate':name,'omission':'atlas','omitted':a,'n_records':int(m.sum()),
                'pooled_r':float(corr(x[m],y[m])),'centered_r':float(corr(center(x[m],g[m]),center(y[m],g[m])))})
        for i in range(len(df)):
            m=np.arange(len(df))!=i;omit.append({'covariate':name,'omission':'type','omitted':g[i]+' / '+df.type.iloc[i],'n_records':int(m.sum()),
                'pooled_r':float(corr(x[m],y[m])),'centered_r':float(corr(center(x[m],g[m]),center(y[m],g[m])))})
    pd.DataFrame(omit).to_csv(EVIDENCE/'omission-sensitivity.csv',index=False)
    mix=[]
    for ip in range(21):
        for ih in range(21-ip):
            weights=np.array([ip,ih,20-ip-ih])/20
            w=np.array([weights[ORDER.index(a)]/sum(g==a) for a in g]);order=np.argsort(y,kind='stable')
            median=y[order][np.searchsorted(np.cumsum(w[order]),.5,side='left')]
            mix.append({'pulmonary_weight':weights[0],'hepatic_weight':weights[1],'cardiac_weight':weights[2],
                        'weighted_median':float(median),'weighted_mean':float(np.sum(w*y)),
                        'below_chance_fraction':float(np.sum(w*(y<.5)))})
    pd.DataFrame(mix).to_csv(EVIDENCE/'atlas-weight-sensitivity.csv',index=False)
    summary={'seed':20261016,'replicates':20000,'sampling_unit':'atlas-type record, within-atlas replacement; conditional descriptive intervals',
             'paired_difference_definition':'r pooled minus r centered, recomputed jointly in the same replicate; not subtraction of independent intervals',
             'covariates':results,'type_medians':df.groupby('atlas').auroc.median().to_dict(),
             'pooled_type_median':float(df.auroc.median()),'pooled_type_mean':float(df.auroc.mean()),
             'equal_atlas_mean_of_type_means':float(df.groupby('atlas').auroc.mean().mean()),
             'equal_atlas_median_of_type_medians':float(df.groupby('atlas').auroc.median().median())}
    write_json('aggregation-extension.json',summary)

def predictions():
    rows=[];ops=[];controls=[];cal=[];prevalence=[];summary={}
    for folder in sorted((ROOT/'experiments').glob('SC*')):
        source=[]
        for p in sorted((folder/'rows').glob('*.json')):
            d=read(p)
            if d['status']!='SCORED':continue
            source.append(d);key={k:d[k] for k in ['row_id','atlas','type','test_donor','calibration_donor','n_train','n_calibration','n_known','n_novel']}
            key['atlas']=ATLAS[key['atlas']]
            for method,s in d['scores'].items():
                rows.append({**key,'method':method,'auroc':s['auroc'],'average_precision':s['average_precision'],'novel_prevalence':s['novel_prevalence'],'temperature':d['temperature']})
                for op in s['operating_points']:
                    rec={**key,'method':method,**{k:v for k,v in op.items() if k!='prevalence_sensitivity'}};ops.append(rec)
                    for prev in op['prevalence_sensitivity']: prevalence.append({**key,'method':method,'nominal_known_coverage':op['nominal_known_coverage'],**prev})
            for temperature in ['raw','temperature']:
                c=d['known_test_calibration_'+temperature]
                cal.append({**key,'calibration':temperature,'temperature':d['temperature'],**{k:v for k,v in c.items() if k!='bins'}})
            if 'permuted_training_labels' in d:
                ctrl=d['permuted_training_labels']
                for method,s in ctrl['scores'].items():controls.append({**key,'method':method,'control_auroc':s['auroc'],'observed_auroc':d['scores'][method]['auroc'],'converged':ctrl['readout_converged']})
        if source:
            summary[folder.name]={'scored_rows':len(source),'target_types':len(set(x['type'] for x in source)),
                                 'held_donors':len(set(x['test_donor'] for x in source)),
                                 'control_rows':sum('permuted_training_labels' in r for r in source)}
    for name,data in [('new-score-comparisons.csv',rows),('new-locked-thresholds.csv',ops),('new-label-controls.csv',controls),
                      ('new-known-calibration.csv',cal),('new-prevalence-sensitivity.csv',prevalence)]:
        pd.DataFrame(data).to_csv(EVIDENCE/name,index=False)
    write_json('new-experiment-summary.json',summary)

if __name__=='__main__':
    normalise();aggregation();predictions();write_json('analysis-source-hashes.json',SOURCES)
    print(json.dumps({'historical_sources':len(SOURCES),'normalized_and_extended':True}))
