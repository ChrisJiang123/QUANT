"""Optional modality contracts. Masked modalities cannot manufacture information."""
import torch
from torch import nn

class SentimentEncoder(nn.Module):
    def __init__(self,input_dim=4,hidden=24):super().__init__();self.encoder=nn.Sequential(nn.Linear(input_dim,hidden),nn.GELU())
    def forward(self,values,mask):
        safe=torch.where(mask[...,None],values,torch.zeros_like(values))
        encoded=self.encoder(safe)*mask[...,None]
        return encoded.sum(dim=1)/mask.sum(dim=1,keepdim=True).clamp_min(1)

class GraphEncoder(nn.Module):
    def __init__(self,input_dim=8,hidden=24):
        super().__init__();self.project=nn.Linear(input_dim,hidden);self.attention=nn.Linear(2*hidden,1,bias=False)
    def forward(self,nodes,adjacency,node_mask):
        nodes=torch.where(node_mask[...,None],nodes,torch.zeros_like(nodes));h=self.project(nodes)
        n=h.shape[1];left=h[:,:,None,:].expand(-1,-1,n,-1);right=h[:,None,:,:].expand(-1,n,-1,-1)
        scores=torch.nn.functional.leaky_relu(self.attention(torch.cat([left,right],dim=-1)).squeeze(-1))
        allowed=adjacency.bool()&node_mask[:,:,None]&node_mask[:,None,:]
        scores=scores.masked_fill(~allowed,-1e9);weights=torch.softmax(scores,dim=-1)*allowed
        weights=weights/weights.sum(dim=-1,keepdim=True).clamp_min(1e-9)
        return torch.bmm(weights,h)*node_mask[...,None],weights

class DynamicFusion(nn.Module):
    def __init__(self,hidden=24,regimes=3):
        super().__init__();self.cross=nn.MultiheadAttention(hidden,2,batch_first=True);self.regime=nn.Linear(regimes,hidden)
        self.gate=nn.Linear(hidden,1)
    def forward(self,modalities,mask,regime):
        # modalities [batch, modality, hidden]; first (price) is the fallback.
        if not mask[:,0].all():raise ValueError('price modality required for fallback')
        safe=torch.where(mask[...,None],modalities,torch.zeros_like(modalities))
        query=safe[:,:1]+self.regime(regime)[:,None,:]
        cross,_=self.cross(query,safe,safe,key_padding_mask=~mask)
        scores=self.gate(safe+cross).squeeze(-1).masked_fill(~mask,-1e9)
        weights=torch.softmax(scores,dim=-1);fused=(safe*weights[...,None]).sum(dim=1)
        return fused,weights
