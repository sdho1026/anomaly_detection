"""Check mask isolation, adapter gradients and actual frozen-weight invariance."""
import json,tempfile
from pathlib import Path
import torch
from model import ModelConfig,Pretrainer,Classifier,VARIANTS

def run_checks(device='cpu'):
    torch.set_num_threads(2);torch.manual_seed(713)
    cfg=ModelConfig();n=(cfg.length+cfg.patch-1)//cfg.patch
    p=torch.randn(2,n,cfg.patch,device=device);q=p.clone();q[:,8:]+=7
    pre=Pretrainer(cfg,'cnn').to(device).eval()
    with torch.no_grad():
        a=pre.backbone.context_from_patches(p,True);b=pre.backbone.context_from_patches(q,True)
        assert torch.allclose(a[:,:9],b[:,:9],atol=1e-5),'Pretraining leaks future clean patches'
        assert torch.allclose(pre.backbone.embed(p)[:,:8],pre.backbone.embed(q)[:,:8],atol=1e-6)
        context=torch.randn(2,n,64,device=device);query=torch.randn_like(context)
        changed=context.clone();changed[:,8:]+=9
        assert torch.allclose(pre.decoder(query,context)[:,:8],pre.decoder(query,changed)[:,:8],atol=1e-6)
    pre.train();loss=pre.loss(torch.randn(2,2501,device=device)*.004)
    loss.backward();assert torch.isfinite(loss)
    assert pre.backbone.point_valid.sum()==2501
    results={};initial_head=None
    for variant in VARIANTS:
        torch.manual_seed(713)
        model=Classifier(cfg,variant).to(device).eval()
        model.backbone.load_state_dict(pre.backbone.state_dict())
        if initial_head is None:initial_head=model.head[-1].weight.detach().clone()
        else:assert torch.equal(initial_head,model.head[-1].weight),'Unpaired head initialization'
        with torch.no_grad():
            a=model.encode_patches(p);b=model.encode_patches(q)
            if model.bidirectional:
                assert not torch.allclose(a[:,:8],b[:,:8],atol=1e-5),'Bidirectional path cannot see future'
            else:
                assert torch.allclose(a[:,:8],b[:,:8],atol=1e-5),'Causal classifier leaks future'
            # Initially zero-up adapters must not change the original encoder function.
            mask=None if model.bidirectional else model.backbone.causal_mask
            reference=model.backbone.encoder(model.backbone.embed(p)+model.backbone.pe,mask=mask)
            assert torch.allclose(a,reference,atol=1e-5),'Initial adapters are not identity'
        # Pooling against a sample-level reference, excluding padded duration.
        z=torch.randn(2,n,64,device=device)
        expanded=z.repeat_interleave(32,dim=1)[:,:2501];edges=[i*2501//8 for i in range(9)]
        expected=torch.cat([torch.stack([expanded[:,edges[i]:edges[i+1]].mean(1) for i in range(8)],1).flatten(1),z.amax(1)],1)
        assert torch.allclose(model.pool(z),expected,atol=1e-6)
        before={k:v.detach().clone() for k,v in model.backbone.state_dict().items()}
        trainable=[p for p in model.parameters() if p.requires_grad]
        if model.use_adapter:
            assert sum(p.numel() for p in trainable)==2769
            assert all(n.startswith(('adapters.','head.')) for n,p in model.named_parameters() if p.requires_grad)
        opt=torch.optim.AdamW(trainable,lr=.001)
        model.train();x=torch.randn(4,2501,device=device)*.004;y=torch.tensor([0.,1.,0.,1.],device=device)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y)
            assert torch.isfinite(loss);loss.backward()
            if model.use_adapter:
                assert all(p.grad is None for p in model.backbone.parameters())
                assert all(a.up.weight.grad is not None and a.up.weight.grad.abs().sum()>0 for a in model.adapters)
            opt.step()
        if model.use_adapter:
            assert all(torch.equal(before[k],v) for k,v in model.backbone.state_dict().items()),'Frozen backbone changed'
            assert all(a.down.weight.grad is not None and a.down.weight.grad.abs().sum()>0 for a in model.adapters),'Gradient lost through later frozen layer'
        else:
            assert not torch.equal(before['embed.proj.weight'],model.backbone.embed.proj.weight),'Full tuning did not update embedding'
        model.eval()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'check.pt';torch.save(model.state_dict(),path)
            restored=Classifier(cfg,variant).to(device).eval()
            restored.load_state_dict(torch.load(path,map_location=device,weights_only=True))
            with torch.no_grad():assert torch.allclose(model(x),restored(x),atol=1e-6)
        results[variant]={'parameters':sum(p.numel() for p in model.parameters()),
                          'trainable_parameters':sum(p.numel() for p in trainable),
                          'classification_attention':'bidirectional' if model.bidirectional else 'causal',
                          'pretraining':'causal, self-only decoder, no adapters',
                          'mask_pool_gradient_freeze_checkpoint_checks':'passed'}
    print(json.dumps(results,indent=2));return results

if __name__=='__main__':run_checks('cuda' if torch.cuda.is_available() else 'cpu')
