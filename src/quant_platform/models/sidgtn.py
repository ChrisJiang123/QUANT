import torch
from torch import nn
import torch.nn.functional as F

class SIDGTN(nn.Module):
    def __init__(self,factors=8,hidden=24,dropout=.15,ablation='full'):
        super().__init__(); self.ablation=ablation
        self.price=nn.Sequential(nn.Conv1d(3,hidden,3,padding=2,dilation=2),nn.GELU(),nn.Dropout(dropout),
                                 nn.Conv1d(hidden,hidden,3),nn.GELU(),nn.AdaptiveAvgPool1d(1))
        self.factor=nn.Sequential(nn.Linear(factors,hidden),nn.GELU(),nn.Dropout(dropout))
        self.factor_gate=nn.Linear(factors,hidden)
        self.asset=nn.Embedding(2,hidden)
        self.gate=nn.Linear(hidden*2,2)
        self.head=nn.Linear(hidden,12)
    def forward(self,price,factors,asset_id):
        p=self.price(price.transpose(1,2)).squeeze(-1)
        f=self.factor(factors)*torch.sigmoid(self.factor_gate(factors))
        if self.ablation=='no_price': p=torch.zeros_like(p)
        if self.ablation=='no_factor': f=torch.zeros_like(f)
        gate=torch.softmax(self.gate(torch.cat([p,f],dim=-1)),dim=-1)
        if self.ablation=='no_gate': gate=torch.ones_like(gate)/2
        hidden=gate[:,:1]*p+gate[:,1:]*f+.1*self.asset(asset_id)
        raw=self.head(hidden).reshape(-1,2,6)
        median=raw[:,:,0]
        quantiles=torch.stack([median-F.softplus(raw[:,:,1])*.1,median,median+F.softplus(raw[:,:,2])*.1],dim=-1)
        return {'quantiles':quantiles,'direction_logit':raw[:,:,3],
                'volatility':F.softplus(raw[:,:,4])*.1,'tail':median-F.softplus(raw[:,:,5])*.1,'gate':gate}

def multitask_loss(output,returns,volatility,single_task=False):
    q=torch.tensor([.1,.5,.9],device=returns.device)
    err=returns[:,:,None]-output['quantiles']
    pinball=torch.maximum(q*err,(q-1)*err).mean()
    if single_task:return pinball
    return pinball+.1*F.binary_cross_entropy_with_logits(output['direction_logit'],(returns>0).float())+\
        .1*F.smooth_l1_loss(output['volatility'],volatility)+.1*F.smooth_l1_loss(output['tail'],returns)
