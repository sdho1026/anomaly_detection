"""Three A3 extensions sharing the same causal pretraining within each fold."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,hashlib,json,math,random,sys,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,f1_score,confusion_matrix,roc_curve
from model import ModelConfig,Pretrainer,Classifier,VARIANTS
from data import load_roles,manifest

ROOT=Path(__file__).resolve().parent
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,path)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def savept(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def loadpt(path,device='cpu'):return torch.load(path,map_location=device,weights_only=True)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
def device_for(name):
    if name=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA GPU가 없습니다. Colab 런타임 유형을 T4 GPU로 바꾸세요.')
    return torch.device(name)
def loader(arr,batch,shuffle=False,seed=42):
    x,y,_=arr
    ds=TensorDataset(torch.from_numpy(x),torch.from_numpy(y).float())
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=0,
                      generator=torch.Generator().manual_seed(seed),drop_last=False)
def writecsv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def predictions(path,arr,score,threshold):
    writecsv(path,[dict(file=n,label=int(y),defect_score=float(s),predicted=int(s>=threshold)) for n,y,s in zip(arr[2],arr[1],score)])
def metrics(y,score,threshold=.5):
    pred=score>=threshold;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(auroc=float(roc_auc_score(y,score)),average_precision=float(average_precision_score(y,score)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),f1=float(f1_score(y,pred,zero_division=0)),
                recall=float(tp/max(tp+fn,1)),specificity=float(tn/max(tn+fp,1)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),threshold=float(threshold))
def choose_threshold(y,p):
    u=np.unique(p);candidates=np.r_[0.,(u[:-1]+u[1:])/2,np.nextafter(u[-1],1.),.5]
    return float(max(candidates,key=lambda t:(balanced_accuracy_score(y,p>=t),-abs(t-.5))))
@torch.no_grad()
def infer(model,arr,batch,device,criterion=None):
    model.eval();scores=[];loss=0.;n=0
    for x,y in loader(arr,batch):
        x=x.to(device);y=y.to(device);logit=model(x)
        if not torch.isfinite(logit).all():raise RuntimeError('Non-finite classification output')
        scores.extend(logit.sigmoid().cpu().numpy().tolist())
        if criterion is not None:loss+=float(criterion(logit,y))*len(x)
        n+=len(x)
    return np.array(scores),loss/max(n,1)
@torch.no_grad()
def pre_val(model,arr,batch,device):
    model.eval();total=0.;n=0
    generator=torch.Generator(device=device).manual_seed(987654)
    for x,_ in loader(arr,batch):
        x=x.to(device);v=model.loss(x,generator)
        total+=float(v)*len(x);n+=len(x)
    return total/n
def pretrain(kind,cfg,c,arrays,root,device):
    folder=root/('pretrain_'+kind);folder.mkdir(exist_ok=True)
    bestpath=folder/'best.pt';lastpath=folder/'last.pt';done=folder/'complete.json'
    if done.exists():print('Reuse completed pretraining:',kind,flush=True);return bestpath
    seed_all(c['seed']);model=Pretrainer(cfg,kind).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=c['pretrain_lr'],weight_decay=c['weight_decay'])
    history=[];best=float('inf');bad=0;start=1
    if lastpath.exists():
        state=loadpt(lastpath,device);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        history=state['history'];best=state['best'];bad=state['bad'];start=state['epoch']+1
        print('Resume pretrain',kind,'epoch',start,flush=True)
    if c['pretrain_epochs']==0:
        savept(bestpath,{'backbone':model.backbone.state_dict(),'epoch':0,'cfg':asdict(cfg)})
        dump(done,{'epoch':0,'note':'Random initialization; no pretraining'});return bestpath
    for epoch in range(start,c['pretrain_epochs']+1):
        if bad>=c['pretrain_patience']:break
        seed_all(c['seed']+epoch);model.train();total=0.;n=0;tick=time.time()
        for x,_ in loader(arrays['train'],c['batch_size'],True,c['seed']+epoch):
            x=x.to(device);opt.zero_grad(set_to_none=True);loss=model.loss(x)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite pretraining loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=float(loss.detach())*len(x);n+=len(x)
        val=pre_val(model,arrays['validation'],c['batch_size'],device)
        if not math.isfinite(val):raise RuntimeError('Non-finite pretraining validation loss')
        if val<best-1e-7:
            best=val;bad=0;savept(bestpath,{'backbone':model.backbone.state_dict(),'epoch':epoch,'cfg':asdict(cfg),'val_loss':val})
        else:bad+=1
        history.append(dict(epoch=epoch,train_loss=total/n,val_loss=val))
        savept(lastpath,dict(model=model.state_dict(),optimizer=opt.state_dict(),history=history,best=best,bad=bad,epoch=epoch))
        writecsv(folder/'history.csv',history)
        print(f'PRE {kind:6s} {epoch:03d} train={total/n:.5f} val={val:.5f} best={best:.5f} {time.time()-tick:.1f}s',flush=True)
    dump(done,{'best_val_loss':best,'epochs_run':len(history),'checkpoint':str(bestpath)})
    del model,opt
    if device.type=='cuda':torch.cuda.empty_cache()
    return bestpath

def backbone_digest(model):
    h=hashlib.sha256()
    for n,t in sorted(model.backbone.state_dict().items()):
        h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def finetune(variant,prepath,cfg,c,arrays,root,device):
    folder=root/variant;folder.mkdir(exist_ok=True);summarypath=folder/'validation.json'
    if summarypath.exists():print('Reuse completed fine-tuning:',variant,flush=True);return read(summarypath)
    seed_all(c['seed']+10000);model=Classifier(cfg,variant).to(device)
    model.backbone.load_state_dict(loadpt(prepath,device)['backbone'])
    frozen_before=backbone_digest(model) if model.use_adapter else None
    profile={'variant':variant,'pretraining_checkpoint_sha256':sha(prepath),
             'attention':'bidirectional' if model.bidirectional else 'causal',
             'parameters':sum(p.numel() for p in model.parameters()),
             'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
             'trainable_names':[n for n,p in model.named_parameters() if p.requires_grad],
             'frozen_backbone_sha256_before':frozen_before,
             'backbone_dropout':'remains active during training in all arms; disabled at evaluation'}
    groups=[]
    for module,lr in [(model.backbone,c['encoder_lr']),(model.adapters,c['adapter_lr']),(model.head,c['head_lr'])]:
        params=[p for p in module.parameters() if p.requires_grad]
        if params:groups.append({'params':params,'lr':lr})
    opt=torch.optim.AdamW(groups,weight_decay=c['weight_decay'])
    y=arrays['train'][1];weight=float((y==0).sum()/(y==1).sum())
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device=device))
    history=[];best_auc=-1.;best_loss=float('inf');bad=0;start=1
    bestpath=folder/'best.pt';lastpath=folder/'last.pt'
    if lastpath.exists():
        st=loadpt(lastpath,device);model.load_state_dict(st['model']);opt.load_state_dict(st['optimizer'])
        history=st['history'];best_auc=st['best_auc'];best_loss=st['best_loss'];bad=st['bad'];start=st['epoch']+1
    for epoch in range(start,c['finetune_epochs']+1):
        if bad>=c['finetune_patience']:break
        seed_all(c['seed']+10000+epoch);model.train();total=0.;n=0;tick=time.time()
        for x,yb in loader(arrays['train'],c['batch_size'],True,c['seed']+10000+epoch):
            x=x.to(device);yb=yb.to(device);opt.zero_grad(set_to_none=True);loss=criterion(model(x),yb)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite fine-tuning loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach())*len(x);n+=len(x)
        p,vl=infer(model,arrays['validation'],c['batch_size'],device,criterion)
        auc=float(roc_auc_score(arrays['validation'][1],p));ap=float(average_precision_score(arrays['validation'][1],p))
        improved=auc>best_auc+1e-9 or (abs(auc-best_auc)<=1e-9 and vl<best_loss-1e-7)
        if improved:
            best_auc=auc;best_loss=vl;bad=0
            savept(bestpath,{'model':model.state_dict(),'cfg':asdict(cfg),'variant':variant,'epoch':epoch,'val_auc':auc,'val_loss':vl})
        else:bad+=1
        history.append(dict(epoch=epoch,train_loss=total/n,val_loss=vl,val_auroc=auc,val_ap=ap))
        savept(lastpath,dict(model=model.state_dict(),optimizer=opt.state_dict(),history=history,best_auc=best_auc,best_loss=best_loss,bad=bad,epoch=epoch))
        writecsv(folder/'history.csv',history)
        print(f'FT  {variant:8s} {epoch:03d} train={total/n:.5f} val={vl:.5f} AUROC={auc:.4f} AP={ap:.4f} {time.time()-tick:.1f}s',flush=True)
    st=loadpt(bestpath,device);model.load_state_dict(st['model']);p,vl=infer(model,arrays['validation'],c['batch_size'],device,criterion)
    threshold=choose_threshold(arrays['validation'][1],p)
    if model.use_adapter:
        profile['frozen_backbone_sha256_after']=backbone_digest(model)
        if profile['frozen_backbone_sha256_after']!=frozen_before:raise RuntimeError('Frozen backbone changed')
    dump(folder/'training_profile.json',profile)
    result={'variant':variant,'best_epoch':st['epoch'],'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':profile['trainable_parameters'],'val_loss':vl,**metrics(arrays['validation'][1],p,threshold)}
    predictions(folder/'validation_predictions.csv',arrays['validation'],p,threshold)
    dump(summarypath,result)
    del model,opt
    if device.type=='cuda':torch.cuda.empty_cache()
    return result

def plot_histories(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,len(VARIANTS),figsize=(15,7),squeeze=False)
    for j,v in enumerate(VARIANTS):
        rows=list(csv.DictReader((root/v/'history.csv').open(encoding='utf-8-sig')))
        e=[int(r['epoch']) for r in rows]
        axes[0,j].plot(e,[float(r['train_loss']) for r in rows],label='train weighted BCE')
        axes[0,j].plot(e,[float(r['val_loss']) for r in rows],label='validation weighted BCE')
        axes[0,j].set_title(v);axes[0,j].legend(fontsize=8)
        axes[1,j].plot(e,[float(r['val_auroc']) for r in rows]);axes[1,j].set(xlabel='Epoch',ylabel='Validation AUROC',ylim=(0,1))
    fig.tight_layout();fig.savefig(root/'training_curves.png',dpi=160);plt.close(fig)

def compare(data_path,split_path,root,c,device):
    root.mkdir(parents=True,exist_ok=True)
    configs={v:ModelConfig(length=math.ceil(50001/c['factor']),patch=c['patch'],dim=c['dim'],layers=c['layers'],heads=c['heads'],ff=c['ff'],dropout=c['dropout'],adapter_dim=c['adapter_dim']) for v in VARIANTS}
    evidence=dict(config=c,model_configs={v:asdict(cfg) for v,cfg in configs.items()},split_sha256=sha(split_path),source_sha256={n:sha(ROOT/n) for n in ('model.py','data.py','run.py','cv.py','dataset_manifest.json','fixed_folds.json')})
    if (root/'protocol.json').exists():
        old=read(root/'protocol.json')
        if old!=evidence:raise ValueError('이 결과 폴더의 설정/코드/분할이 다릅니다. 새 RUN_NAME을 사용하세요.')
    else:dump(root/'protocol.json',evidence)
    dump(root/'environment.json',{'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None})
    arrays=load_roles(data_path,split_path,('train','validation'),c['factor'])
    for role,arr in arrays.items():print(role,arr[0].shape,'normal',int((arr[1]==0).sum()),'defect',int((arr[1]==1).sum()),flush=True)
    cfg=configs[VARIANTS[0]]
    path=pretrain('cnn',cfg,c,arrays,root,device)
    results=[finetune(v,path,configs[v],c,arrays,root,device) for v in VARIANTS]
    writecsv(root/'validation_comparison.csv',results)
    winner=max(results,key=lambda r:(r['auroc'],-r['val_loss']))['variant']
    selection={'selected_by_validation':winner,'rule':'max validation AUROC, tie min validation weighted BCE; complete tie CAUSAL_ADAPTER/BI_FULL/BI_ADAPTER order',
               'checkpoint_sha256':{v:sha(root/v/'best.pt') for v in VARIANTS},'validation_file_sha256':{v:sha(root/v/'validation.json') for v in VARIANTS},
               'note':'Frozen BEFORE test. All 144 records were previously explored; this is internal stratified cross-validation.'}
    dump(root/'selection_before_test.json',selection);plot_histories(root)
    print('VALIDATION COMPARISON',json.dumps(results,ensure_ascii=False,indent=2),flush=True)
    print('Selected on validation:',winner,'| No test waveforms loaded in compare stage.',flush=True)

def auc_ci(y,p,n=3000):
    rng=np.random.default_rng(734);a=p[y==0];b=p[y==1]
    pair=(b[:,None]>a).astype(float)+.5*(b[:,None]==a)
    vals=[pair[np.ix_(rng.integers(len(b),size=len(b)),rng.integers(len(a),size=len(a)))].mean() for _ in range(n)]
    return np.quantile(vals,[.025,.975]).tolist()

def final_test(data_path,split_path,root,device):
    protocol=read(root/'protocol.json');c=protocol['config'];selection=read(root/'selection_before_test.json')
    if sha(split_path)!=protocol['split_sha256']:raise ValueError('Split changed')
    for n,h in protocol['source_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Code changed after training: '+n)
    for v in VARIANTS:
        if sha(root/v/'best.pt')!=selection['checkpoint_sha256'][v] or sha(root/v/'validation.json')!=selection['validation_file_sha256'][v]:raise ValueError('Checkpoint or validation selection changed')
    if (root/'test_comparison.csv').exists():
        print('Already evaluated: reusing saved test results.');print((root/'test_comparison.csv').read_text(encoding='utf-8-sig'));return
    # First test decoding occurs here, after all model selection is frozen.
    test=load_roles(data_path,split_path,('test',),c['factor'])['test']
    results=[];curves={}
    for v in VARIANTS:
        cfg=ModelConfig(**protocol['model_configs'][v])
        model=Classifier(cfg,v).to(device);model.load_state_dict(loadpt(root/v/'best.pt',device)['model'])
        threshold=read(root/v/'validation.json')['threshold'];p,_=infer(model,test,c['batch_size'],device)
        r={'variant':v,'selected_by_validation':v==selection['selected_by_validation'],**metrics(test[1],p,threshold)}
        lo,hi=auc_ci(test[1],p);r['auroc_bootstrap_low']=lo;r['auroc_bootstrap_high']=hi
        predictions(root/v/'test_predictions.csv',test,p,threshold);dump(root/v/'test.json',r);results.append(r);curves[v]=p
        del model
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,5))
    for r in results:
        fpr,tpr,_=roc_curve(test[1],curves[r['variant']]);ax.plot(fpr,tpr,label=f"{r['variant']}: {r['auroc']:.3f}")
    ax.plot([0,1],[0,1],'k--',alpha=.4);ax.legend();ax.set(xlabel='False positive rate',ylabel='True positive rate',title=f"Fold test: {int((test[1]==0).sum())} normal + {int((test[1]==1).sum())} defect")
    fig.tight_layout();fig.savefig(root/'test_roc.png',dpi=160);plt.close(fig)
    dump(root/'test_notes.json',{'selected_model':selection['selected_by_validation'],'test_counts':{'normal':int((test[1]==0).sum()),'defect':int((test[1]==1).sum())},
                              'CI':'Stratified fixed-model bootstrap; few defects per fold; ignores dependence, training variability and previous data exploration.',
                              'interpretation':'Compare prespecified arms. Do not select the best test arm or retune using these scores. AUROC >= 0.85 is not guaranteed.'})
    writecsv(root/'test_comparison.csv',results)
    print('FINAL TEST',json.dumps(results,ensure_ascii=False,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['compare','test']);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--split',required=True);p.add_argument('--config',default=str(ROOT/'config.json'));p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads);device=device_for(a.device)
    if a.stage=='compare':compare(a.data,a.split,Path(a.out),read(a.config),device)
    else:final_test(a.data,a.split,Path(a.out),device)
if __name__=='__main__':main()
