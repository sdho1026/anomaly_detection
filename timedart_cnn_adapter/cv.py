"""Prespecified 5-fold outer evaluation with one stratified inner holdout."""
import argparse, csv, json, shutil
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from data import manifest
from run import ROOT, VARIANTS, dump, read, sha, writecsv, compare, final_test, device_for

SOURCES=('model.py','data.py','run.py','cv.py','dataset_manifest.json','fixed_folds.json','checks.py')
METRICS=('auroc','average_precision','balanced_accuracy','f1','recall','specificity')

def make_splits():
    # Freeze the exact folds from the user's completed A/B experiment.
    docs=read(ROOT/'fixed_folds.json');assert len(docs)==5
    reference={r['file'] for r in read(ROOT/'dataset_manifest.json')['records']}
    all_test=[]
    for fold,doc in enumerate(docs,1):
        assert doc['fold']==fold
        roles={k:{r['file'] for r in v} for k,v in doc['splits'].items()}
        assert not (roles['train']&roles['validation'] or roles['train']&roles['test'] or roles['validation']&roles['test'])
        assert set.union(*roles.values())==reference
        assert all({r['label'] for r in rows}=={0,1} for rows in doc['splits'].values())
        all_test.extend(roles['test'])
    assert Counter(all_test)==Counter(reference)
    return docs

def prepare(data_path,root,c):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if c.get('patch')!=32 or c.get('adapter_dim')!=8:raise ValueError('This comparison fixes patch32 and adapter bottleneck8')
    docs=make_splits()
    evidence={'config':c,'n_splits':5,'outer_seed':42,'inner_val_fraction':.20,
              'inner_seeds':[1001,1002,1003,1004,1005],
              'training_seed_rule':'same config.seed for all folds; fresh initialization in each fold',
              'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'folds':docs}
    guard=root/'cv_protocol.json'
    if guard.exists() and read(guard)!=evidence:
        raise ValueError('Code/config/folds changed. Use a new RUN_NAME; do not reuse old checkpoints.')
    dump(guard,evidence);splitdir=root/'splits';splitdir.mkdir(exist_ok=True)
    counts=[]
    for doc in docs:
        fold=doc['fold'];path=splitdir/f'fold_{fold:02d}.json'
        if path.exists() and read(path)!=doc:raise ValueError('Saved fold manifest changed')
        dump(path,doc);manifest(data_path,path)
        for role,rows in doc['splits'].items():
            counts.append({'fold':fold,'role':role,'n':len(rows),
                           'normal':sum(r['label']==0 for r in rows),'defect':sum(r['label']==1 for r in rows)})
    writecsv(root/'fold_counts.csv',counts)
    snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for n in (*SOURCES,'README_KO.md'):
        shutil.copy2(ROOT/n,snapshot/n)
    dump(snapshot/'config.json',c)
    return docs

def verify_run(data_path,root):
    p=read(root/'cv_protocol.json')
    for n,h in p['source_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Code changed: '+n)
    for doc in p['folds']:
        split=root/'splits'/f"fold_{doc['fold']:02d}.json"
        if read(split)!=doc:raise ValueError('Fold changed')
        manifest(data_path,split)
        folder=root/f"fold_{doc['fold']:02d}"
        protocol=read(folder/'protocol.json')
        if protocol['config']!=p['config'] or protocol['split_sha256']!=sha(split):
            raise ValueError('Training protocol mismatch')
        selection=read(folder/'selection_before_test.json')
        for v in VARIANTS:
            if sha(folder/v/'best.pt')!=selection['checkpoint_sha256'][v]:raise ValueError('Checkpoint changed')
            if sha(folder/v/'validation.json')!=selection['validation_file_sha256'][v]:raise ValueError('Validation changed')
    return p

def summarize(root):
    rows=[];selected=[]
    for fold in range(1,6):
        folder=root/f'fold_{fold:02d}'
        test_rows=list(csv.DictReader((folder/'test_comparison.csv').open(encoding='utf-8-sig')))
        if {r['variant'] for r in test_rows}!=set(VARIANTS) or len(test_rows)!=len(VARIANTS):raise ValueError('Missing model result')
        winner=read(folder/'selection_before_test.json')['selected_by_validation']
        for r in test_rows:
            r={k:(float(v) if k not in ('variant','selected_by_validation') else v) for k,v in r.items()}
            r['fold']=fold;r['selected_by_validation']=r['variant']==winner
            rows.append(r)
            if r['variant']==winner:selected.append(dict(r))
    writecsv(root/'cv_fold_metrics.csv',rows)
    writecsv(root/'cv_inner_selected_fold_metrics.csv',selected)
    def aggregate(group,name):
        out={'variant':name,'n_folds':len(group)}
        for metric in METRICS:
            a=np.array([r[metric] for r in group])
            out[metric+'_mean']=float(a.mean());out[metric+'_std']=float(a.std(ddof=1))
        return out
    summary=[aggregate([r for r in rows if r['variant']==v],v) for v in VARIANTS]
    writecsv(root/'cv_summary.csv',summary)
    writecsv(root/'cv_inner_selected_summary.csv',[aggregate(selected,'inner_validation_selection')])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.5))
    for v in VARIANTS:
        group=[r for r in rows if r['variant']==v]
        ax.plot([r['fold'] for r in group],[r['auroc'] for r in group],marker='o',label=v)
    ax.axhline(.5,color='gray',linestyle='--',linewidth=1)
    ax.axhline(.85,color='gray',linestyle=':',linewidth=1)
    ax.set(xlabel='Outer test fold',ylabel='AUROC',ylim=(0,1),xticks=range(1,6),title='Prespecified models: stratified 5-fold')
    ax.legend();fig.tight_layout();fig.savefig(root/'cv_auroc.png',dpi=160);plt.close(fig)
    dump(root/'cv_notes.json',{
        'summary':'Unweighted mean and sample standard deviation (ddof=1) across five outer folds. SD is not a confidence interval.',
        'selected_policy':'Separate summary for a policy choosing among CAUSAL_ADAPTER/BI_FULL/BI_ADAPTER using only the inner validation AUROC (tie: validation loss).',
        'limitations':'Only 18 defects; each outer test has 3 or 4. Overlapping training sets make folds dependent. All records were previously explored; internal resampling, not an untouched external test.',
        'groups':'Stratification is by label, not specimen/session/location. This does not establish generalization to new specimens or sessions.',
        'selection':'Architectures/config are fixed before running. Do not select settings using outer scores and call those scores unbiased performance.',
        'predictions':'Per-fold prediction files are retained for audit. No pooled OOF score or pooled OOF AUROC is computed.',
        'target':'AUROC >= 0.85 is not guaranteed.'})
    print(json.dumps(summary,indent=2),flush=True)
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=('prepare','train','test','summary'))
    ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--config',default=str(ROOT/'config.json'))
    ap.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    ap.add_argument('--threads',type=int,default=2)
    a=ap.parse_args();root=Path(a.out);torch.set_num_threads(a.threads)
    if a.stage in ('prepare','train'):
        c=read(a.config);prepare(a.data,root,c)
        if a.stage=='prepare':return
        device=device_for(a.device)
        for fold in range(1,6):
            print(f'\n===== TRAIN FOLD {fold}/5 =====',flush=True)
            compare(a.data,root/'splits'/f'fold_{fold:02d}.json',root/f'fold_{fold:02d}',c,device)
    else:
        verify_run(a.data,root)
        if a.stage=='test':
            device=device_for(a.device)
            for fold in range(1,6):
                print(f'\n===== TEST FOLD {fold}/5 =====',flush=True)
                final_test(a.data,root/'splits'/f'fold_{fold:02d}.json',root/f'fold_{fold:02d}',device)
        summarize(root)

if __name__=='__main__':main()
