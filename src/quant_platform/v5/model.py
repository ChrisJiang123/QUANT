"""Five-horizon SIDGTN with real masked modality paths and a gated fusion upgrade."""
import math
import torch
from torch import nn
import torch.nn.functional as F

HORIZONS=(1,3,7,14,28)

class V5SIDGTN(nn.Module):
    def __init__(self,hidden=24,lookback=60,factors=8,aux=8,nodes=8,dropout=.15,
                 variant='core',dynamic_fusion=False,fusion_qualified=False):
        super().__init__()
        if dynamic_fusion and not fusion_qualified:raise PermissionError('dynamic fusion lacks registered incremental evidence')
        if variant not in {'core','factor','graph','sentiment'}:raise ValueError('unknown registered variant')
        self.variant=variant;self.dynamic_fusion=dynamic_fusion
        self.constructor=dict(hidden=hidden,lookback=lookback,factors=factors,aux=aux,nodes=nodes,dropout=dropout,
                              variant=variant,dynamic_fusion=dynamic_fusion,fusion_qualified=fusion_qualified)
        self.price_conv=nn.Conv1d(3,hidden,3,padding=1)
        self.position=nn.Parameter(torch.randn(1,lookback,hidden)*.02)
        layer=nn.TransformerEncoderLayer(hidden,2,hidden*2,dropout,batch_first=True,norm_first=True)
        self.price_encoder=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        self.factor=nn.Sequential(nn.Linear(factors,hidden),nn.GELU(),nn.Dropout(dropout))
        self.factor_gate=nn.Linear(factors,hidden)
        self.auxiliary=nn.Sequential(nn.Linear(aux*2,hidden),nn.GELU(),nn.Dropout(dropout))
        self.node_project=nn.Linear(factors,hidden)
        self.node_id=nn.Embedding(nodes,hidden)
        self.edge_attention=nn.Linear(2*hidden,1,bias=False)
        self.node_pool=nn.Linear(hidden,1,bias=False)
        self.sentiment=nn.Sequential(nn.Linear(4,hidden),nn.GELU(),nn.Dropout(dropout))
        self.asset=nn.Embedding(2,hidden)
        self.static_logits=nn.Parameter(torch.zeros(5))
        self.cross=nn.MultiheadAttention(hidden,2,dropout=dropout,batch_first=True)
        self.regime=nn.Linear(3,hidden)
        self.dynamic_gate=nn.Linear(hidden,1)
        self.head=nn.Linear(hidden,len(HORIZONS)*6)
        self.register_buffer('temperature',torch.ones(len(HORIZONS)))
        self.register_buffer('horizons',torch.tensor(HORIZONS,dtype=torch.float32))

    def forward(self,batch):
        price=batch['price'];x=batch['factors'];b=len(x)
        temporal=self.price_conv(price.transpose(1,2)).transpose(1,2)+self.position[:,:price.shape[1]]
        p=self.price_encoder(temporal).mean(dim=1)
        f=self.factor(x)*torch.sigmoid(self.factor_gate(x))
        am=batch['aux_mask'].bool();av=torch.where(am,batch['aux'],torch.zeros_like(batch['aux']))
        aux=self.auxiliary(torch.cat([av,am.to(av.dtype)],dim=-1))
        nm=batch['node_mask'].bool();nv=torch.where(nm[...,None],batch['nodes'],torch.zeros_like(batch['nodes']))
        h=self.node_project(nv)+self.node_id.weight[None,:,:]
        n=h.shape[1];left=h[:,:,None,:].expand(-1,-1,n,-1);right=h[:,None,:,:].expand(-1,n,-1,-1)
        allowed=batch['adjacency'].bool()&nm[:,:,None]&nm[:,None,:]
        scores=F.leaky_relu(self.edge_attention(torch.cat([left,right],dim=-1)).squeeze(-1),.2)
        attention=torch.softmax(scores.masked_fill(~allowed,-1e9),dim=-1)*allowed
        attention=attention/attention.sum(dim=-1,keepdim=True).clamp_min(1e-8)
        gh=torch.bmm(attention,h)*nm[...,None]
        pool=torch.softmax(self.node_pool(gh).squeeze(-1).masked_fill(~nm,-1e9),dim=-1)*nm
        pool=pool/pool.sum(dim=-1,keepdim=True).clamp_min(1e-8)
        graph=(gh*pool[...,None]).sum(dim=1)
        sm=batch['sentiment_mask'].bool()
        sv=torch.where(sm[...,None],batch['sentiment'],torch.zeros_like(batch['sentiment']))
        sent=(self.sentiment(sv)*sm[...,None]).sum(dim=1)/sm.sum(dim=1,keepdim=True).clamp_min(1)
        presence=torch.stack([torch.ones(b,device=x.device,dtype=torch.bool),torch.ones(b,device=x.device,dtype=torch.bool),
            am.any(dim=1)&(self.variant in {'factor','graph','sentiment'}),nm.any(dim=1)&(self.variant in {'graph','sentiment'}),
            sm.any(dim=1)&(self.variant=='sentiment')],dim=1)
        modalities=torch.stack([p,f,aux,graph,sent],dim=1)
        safe=torch.where(presence[...,None],modalities,torch.zeros_like(modalities))
        if self.dynamic_fusion:
            regime=batch.get('regime',torch.zeros((b,3),device=x.device,dtype=x.dtype))
            q=safe[:,:1]+self.regime(regime)[:,None,:]
            cross,_=self.cross(q,safe,safe,key_padding_mask=~presence)
            logits=self.dynamic_gate(safe+cross).squeeze(-1)
        else:logits=self.static_logits[None,:].expand(b,-1)
        weights=torch.softmax(logits.masked_fill(~presence,-1e9),dim=1)
        hidden=(safe*weights[...,None]).sum(dim=1)+.1*self.asset(batch['asset_id'].long())
        raw=self.head(hidden).reshape(b,len(HORIZONS),6)
        median=raw[:,:,0]
        nq=torch.stack([median-F.softplus(raw[:,:,1]),median,median+F.softplus(raw[:,:,2])],dim=-1)
        anchor=batch['annual_anchor'].clamp_min(.10)[:,None]
        scale=anchor*torch.sqrt(self.horizons[None,:]/365.)
        quantiles=nq*scale[:,:,None]
        volatility=.10+F.softplus(raw[:,:,4])*anchor
        tail=(nq[:,:,0]-F.softplus(raw[:,:,5]))*scale
        probability=torch.sigmoid(raw[:,:,3]/self.temperature.clamp(.5,4.))
        return dict(quantiles=quantiles,normalized_quantiles=nq,direction_logit=raw[:,:,3],
                    direction_score=2*probability-1,direction_probability=probability,
                    volatility=volatility,tail=tail,normalized_tail=tail/scale,
                    modality_mask=presence,fusion_weights=weights,graph_attention=attention,graph_pool=pool)

def loss(model,batch,weights):
    output=model(batch);mask=batch['label_mask'].float()
    hw=torch.tensor(weights['horizon_weights'],device=mask.device)[None,:]*mask
    anchor=batch['annual_anchor'].clamp_min(.10)[:,None]
    scale=anchor*torch.sqrt(model.horizons[None,:]/365.)
    target=batch['target']/scale
    qs=torch.tensor([.1,.5,.9],device=mask.device)
    err=target[:,:,None]-output['normalized_quantiles']
    pin=torch.maximum(qs*err,(qs-1)*err).mean(dim=-1)
    direction=F.binary_cross_entropy_with_logits(output['direction_logit'],(batch['target']>0).float(),reduction='none')
    vol=F.smooth_l1_loss(output['volatility']/anchor,batch['target_vol']/anchor,reduction='none')
    error=target-output['normalized_tail'];tail=torch.maximum(.05*error,-.95*error)
    terms=dict(quantile=pin,direction=direction,annual_vol_normalized_huber=vol,tail_q05_pinball=tail)
    total=sum(weights[k]*(v*hw).sum()/hw.sum().clamp_min(1e-9) for k,v in terms.items())
    return total,output,{k:float((v.detach()*hw).sum()/hw.sum().clamp_min(1e-9)) for k,v in terms.items()}
