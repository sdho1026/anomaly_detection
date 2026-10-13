"""Compact paper-based TimeDART adaptation, not a byte-for-byte official port.
Reference: https://arxiv.org/html/2410.05711v3 (sections 3.1-3.3).
"""
import math
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class ModelConfig:
    length: int = 2501
    patch: int = 32
    dim: int = 64
    layers: int = 2
    heads: int = 4
    ff: int = 128
    dropout: float = 0.1
    diffusion_steps: int = 1000
    adapter_dim: int = 8

class PatchEmbedding(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__(); self.kind=kind; self.cfg=cfg
        if kind=='linear':
            self.proj=nn.Linear(cfg.patch,cfg.dim)
        elif kind=='cnn':
            self.conv3=nn.Conv1d(1,8,3,padding=1)
            self.conv5=nn.Conv1d(1,8,5,padding=2)
            self.proj=nn.Linear(16*cfg.patch,cfg.dim)
        else: raise ValueError(kind)
    def forward(self,p):
        # Each B*N patch becomes a separate batch item: convolutions cannot cross patches.
        if self.kind=='linear': return self.proj(p)
        b,n,l=p.shape; q=p.reshape(b*n,1,l)
        q=torch.cat([F.gelu(self.conv3(q)),F.gelu(self.conv5(q))],dim=1)
        return self.proj(q.flatten(1)).reshape(b,n,-1)

class Backbone(nn.Module):
    def __init__(self,cfg,kind):
        super().__init__(); self.cfg=cfg; self.kind=kind
        # Construct common modules before variant-specific embedding for paired initialization.
        layer=nn.TransformerEncoderLayer(cfg.dim,cfg.heads,cfg.ff,cfg.dropout,
                                         activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,cfg.layers,norm=nn.LayerNorm(cfg.dim),enable_nested_tensor=False)
        self.sos=nn.Parameter(torch.zeros(1,1,cfg.dim));nn.init.normal_(self.sos,std=.02)
        self.embed=PatchEmbedding(cfg,kind)
        n=math.ceil(cfg.length/cfg.patch)
        pos=torch.arange(n,dtype=torch.float32).unsqueeze(1)
        div=torch.exp(torch.arange(0,cfg.dim,2,dtype=torch.float32)*(-math.log(10000)/cfg.dim))
        pe=torch.zeros(n,cfg.dim);pe[:,0::2]=torch.sin(pos*div);pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
        self.register_buffer('causal_mask',torch.ones(n,n,dtype=torch.bool).triu(1))
        weights=torch.ones(n*cfg.patch);weights[cfg.length:]=0
        self.register_buffer('point_valid',weights.reshape(n,cfg.patch))
    def patchify(self,x):
        if x.ndim!=2 or x.shape[1]!=self.cfg.length:raise ValueError(f'Expected [B,{self.cfg.length}], got {tuple(x.shape)}')
        # Paper-style per-scan instance normalization; never fitted across records.
        mu=x.mean(1,keepdim=True);sd=x.std(1,keepdim=True,unbiased=False).clamp_min(1e-8)
        q=(x-mu)/sd
        q=F.pad(q,(0,(-q.shape[1])%self.cfg.patch),mode='replicate')
        return q.reshape(len(x),-1,self.cfg.patch)
    def context_from_patches(self,p,shifted):
        z=self.embed(p)
        if shifted:z=torch.cat([self.sos.expand(len(p),-1,-1),z[:,:-1]],dim=1)
        return self.encoder(z+self.pe,mask=self.causal_mask)
    def forward(self,x):return self.context_from_patches(self.patchify(x),shifted=False)

class Denoiser(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cross=nn.MultiheadAttention(cfg.dim,cfg.heads,dropout=cfg.dropout,batch_first=True)
        self.n1=nn.LayerNorm(cfg.dim);self.n2=nn.LayerNorm(cfg.dim)
        self.ff=nn.Sequential(nn.Linear(cfg.dim,cfg.ff),nn.GELU(),nn.Dropout(cfg.dropout),nn.Linear(cfg.ff,cfg.dim))
        self.drop=nn.Dropout(cfg.dropout);self.out=nn.Linear(cfg.dim,cfg.patch)
    def forward(self,q,context):
        b,n,d=q.shape
        # Length-one attention per patch is algebraically diagonal/self-only cross-attention.
        q=q.reshape(b*n,1,d);c=context.reshape(b*n,1,d)
        a,_=self.cross(q,c,c,need_weights=False)
        h=self.n1(q+self.drop(a));h=self.n2(h+self.drop(self.ff(h)))
        return self.out(h).reshape(b,n,-1)

class Pretrainer(nn.Module):
    def __init__(self,cfg,kind):
        super().__init__();self.cfg=cfg;self.backbone=Backbone(cfg,kind);self.decoder=Denoiser(cfg)
        t=torch.linspace(0,cfg.diffusion_steps,cfg.diffusion_steps+1,dtype=torch.float64)
        abar=torch.cos(((t/cfg.diffusion_steps+.008)/1.008)*math.pi/2)**2
        abar=abar/abar[0];betas=(1-abar[1:]/abar[:-1]).clamp(0,.999)
        self.register_buffer('abar',torch.cumprod(1-betas,0).float())
    def loss(self,x,generator=None):
        clean=self.backbone.patchify(x)
        ctx=self.backbone.context_from_patches(clean,shifted=True)
        steps=torch.randint(self.cfg.diffusion_steps,clean.shape[:2],device=x.device,generator=generator)
        eps=torch.randn(clean.shape,device=x.device,dtype=clean.dtype,generator=generator)
        a=self.abar[steps].unsqueeze(-1)
        noisy=a.sqrt()*clean+(1-a).sqrt()*eps
        query=self.backbone.embed(noisy)+self.backbone.pe
        pred=self.decoder(query,ctx)
        valid=self.backbone.point_valid
        return ((pred-clean).square()*valid).sum()/(len(x)*valid.sum())

VARIANTS=('CAUSAL_ADAPTER','BI_FULL','BI_ADAPTER')

class Adapter(nn.Module):
    def __init__(self,dim,bottleneck):
        super().__init__()
        self.down=nn.Linear(dim,bottleneck)
        self.up=nn.Linear(bottleneck,dim)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def forward(self,h):return h+self.up(F.gelu(self.down(h)))

class Classifier(nn.Module):
    """Eight ordered time-bin means + global max, followed by one linear head."""
    def __init__(self,cfg,variant,amp_mean=None,amp_std=None):
        super().__init__();self.variant=variant;self.cfg=cfg
        if variant not in VARIANTS:raise ValueError(variant)
        self.use_adapter=variant!='BI_FULL'
        self.bidirectional=variant!='CAUSAL_ADAPTER'
        self.backbone=Backbone(cfg,'cnn')
        self.head=nn.Sequential(nn.Dropout(cfg.dropout),nn.Linear(9*cfg.dim,1))
        # Create the identical head before adapter initialization for matched head seeds.
        self.adapters=nn.ModuleList([Adapter(cfg.dim,cfg.adapter_dim) for _ in range(cfg.layers)] if self.use_adapter else [])
        if self.use_adapter:
            self.backbone.requires_grad_(False)
        # Weight each token by its REAL sample overlap with each time bin.
        # Padding contributes no additional duration. The partly valid last token remains.
        n=math.ceil(cfg.length/cfg.patch)
        edges=torch.arange(9,dtype=torch.int64)*cfg.length//8
        start=torch.arange(n,dtype=torch.int64)*cfg.patch
        end=(start+cfg.patch).clamp(max=cfg.length)
        overlap=(torch.minimum(edges[1:,None],end[None,:])-torch.maximum(edges[:-1,None],start[None,:])).clamp_min(0).float()
        if (overlap.sum(1)==0).any():raise ValueError('Input length must be at least 8')
        self.register_buffer('bin_weights',overlap/overlap.sum(1,keepdim=True))
        self.register_buffer('token_valid',end>start)
    def pool(self,z):
        regional=torch.einsum('bnd,kn->bkd',z,self.bin_weights).flatten(1)
        global_max=z.masked_fill(~self.token_valid[None,:,None],float('-inf')).amax(1)
        return torch.cat([regional,global_max],dim=1)
    def encode_patches(self,p):
        # Classification only: UNshifted input. Pretrainer continues using the original
        # shifted causal Backbone path, with no adapters and no mask removal.
        z=self.backbone.embed(p)+self.backbone.pe
        mask=None if self.bidirectional else self.backbone.causal_mask
        for i,layer in enumerate(self.backbone.encoder.layers):
            # Never use no_grad here: adapter gradients must pass through later frozen layers.
            z=layer(z,src_mask=mask)
            if self.use_adapter:z=self.adapters[i](z)
        return self.backbone.encoder.norm(z)
    def forward(self,x):
        z=self.encode_patches(self.backbone.patchify(x))
        return self.head(self.pool(z)).squeeze(-1)
