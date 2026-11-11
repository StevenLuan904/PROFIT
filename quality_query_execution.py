"""Evaluate specified PROFIT outputs using their complete original sensor context."""
from types import MethodType
import torch
class QualityQueryExecution:
 def __init__(self,model,queries,candidate_chunk=32):
  if model.training:raise ValueError('Frozen BatchNorm/eval is required')
  if any(m.training for m in model.modules()):raise ValueError('All modules must be in eval mode')
  if any(getattr(c,'individual',False) for c in model.component_models):raise ValueError('Shared original readout is required')
  self.model=model;self.queries=tuple(int(i) for i in queries);self.chunk=int(candidate_chunk);self.patches=[];self.last_phase=None
  if not self.queries or len(set(self.queries))!=len(self.queries):raise ValueError('Distinct requested outputs required')
 def __enter__(self):
  q=self.queries
  self.query_tensor=torch.tensor(q,dtype=torch.long,device=next(self.model.parameters()).device)
  for component in self.model.component_models:
   head=component.head
   adapter=getattr(head,'adapter',None)
   if adapter is not None:
    if 'forward' in adapter.__dict__:raise ValueError('An existing adapter forward override is active')
    def forward(this,encoded):
     # Keep the original mean over ALL trained channel factors.
     latent=this.down(encoded);centered=this.up-this.up.mean(dim=0,keepdim=True)
     indices=self.query_tensor
     return torch.einsum('bcr,chr->bch',latent,centered.index_select(0,indices))*this.scale
    adapter.forward=MethodType(forward,adapter);self.patches.append(adapter)
  return self
 def phase(self,x,queries):
  B,C,L=x.shape;K=self.model.k_leaders
  if K==0:
   empty=torch.empty(B,len(queries),0,device=x.device,dtype=torch.long)
   return empty,empty.clone(),x.new_empty(B,len(queries),0)
  spectrum=torch.fft.rfft(x,dim=-1);norm=spectrum/spectrum.abs().clamp(min=1e-8)
  target=norm.index_select(1,queries);conj=norm.conj();values=[];lags=[]
  for start in range(0,C,self.chunk):
   cross=target.unsqueeze(2)*conj[:,start:start+self.chunk].unsqueeze(1)
   peak,index=torch.fft.irfft(cross,n=L,dim=-1).max(dim=-1)
   values.append(peak);lags.append(torch.where(index>L//2,index-L,index))
  values=torch.cat(values,-1);lags=torch.cat(lags,-1)
  # The entire original candidate axis and torch.topk tie rule are retained.
  mask=queries[:,None]==torch.arange(C,device=x.device)[None,:]
  values=values-2*mask.to(values.dtype).unsqueeze(0)
  score,ids=values.topk(K,dim=-1)
  return ids,lags.gather(-1,ids),score
 def __call__(self,input):
  m=self.model
  if m.training:raise ValueError('Inference only')
  B,L,C=input.shape;q=self.query_tensor
  if min(self.queries)<0 or max(self.queries)>=C:raise ValueError('Output index outside observed sensors')
  x=m.revin_layer(input,'norm') if m.revin else input
  x=x.permute(0,2,1);original=x.index_select(1,q)
  with torch.no_grad():ids,lag,score=self.phase(x,q)
  self.last_phase=(ids,lag,score)
  if m.k_leaders:
   indices=torch.clamp(torch.arange(L,device=x.device).view(1,1,L,1)-lag.unsqueeze(2),0,L-1)
   selected=x[torch.arange(B,device=x.device).view(B,1,1,1),ids.unsqueeze(2),indices]
   selected=selected*(1-2*(score<0).to(x.dtype).unsqueeze(2))
   selected=torch.where((score.abs()<m.corr_threshold).unsqueeze(2),original.unsqueeze(-1),selected)
   expanded=torch.cat([original.unsqueeze(-1),selected],dim=-1)
  else:expanded=original.unsqueeze(-1)
  outputs=[m.component_models[i](band) for i,band in enumerate(m._iter_frequency_components(expanded))]
  output=torch.stack(outputs,dim=-1).sum(dim=-1).permute(0,2,1)
  if m.revin:
   # Reuse the actual original affine inverse and all original saved moments.
   full=output.new_zeros(B,output.shape[1],C);full.index_copy_(2,q,output)
   output=m.revin_layer(full,'denorm').index_select(2,q)
  return output
 def __exit__(self,*args):
  for adapter in self.patches:del adapter.__dict__['forward']
  self.patches=[];self.last_phase=None
