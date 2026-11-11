"""Loss, TRAIN threshold calibration and regularization functions extracted from the experiment implementation."""
import hashlib
import math


def require(ok,message):
    if not ok:raise ValueError(message)

def array_sha(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def left_weighted_median(values,integer_weights):
    """Smallest sorted value whose exact integer cumulative weight >= half."""
    import numpy as np
    values=np.asarray(values,dtype=np.float64);weights=np.asarray(integer_weights)
    require(values.ndim==weights.ndim==1 and len(values)==len(weights)>0 and np.isfinite(values).all(),'Invalid weighted median values')
    require(np.issubdtype(weights.dtype,np.integer) and (weights>0).all(),'Positive exact integer weights required')
    total=sum(int(v) for v in weights);cut=(total+1)//2;order=np.argsort(values,kind='stable')
    if total<=np.iinfo(np.int64).max:
        cumulative=np.cumsum(weights[order],dtype=np.int64);index=int(np.searchsorted(cumulative,cut,side='left'))
    else:
        cumulative=0
        for index,item in enumerate(order):
            cumulative+=int(weights[item])
            if cumulative>=cut:break
    return float(values[order[index]])

def thresholds_from_z(z32,train_end,seq_len=360,horizons=(12,24,36,48)):
    """Only z32[:train_end]; FP64 prefix sums, exact equal total weight per H."""
    import numpy as np
    z=np.asarray(z32);require(z.dtype==np.float32 and z.ndim==2 and 0<seq_len<train_end<=len(z),'Canonical z32 prefix required');z=z[:train_end]
    require(np.isfinite(z).all() and all(type(h) is int and h>0 and train_end-h>=seq_len for h in horizons),'Invalid training horizons')
    prefix=np.concatenate([np.zeros((1,z.shape[1]),dtype=np.float64),np.cumsum(z.astype(np.float64),axis=0,dtype=np.float64)])
    residuals=[];totals=[];index_contract=[]
    for h in horizons:
        origins=np.arange(seq_len,train_end-h+1,dtype=np.int64);mean=(prefix[origins]-prefix[origins-seq_len])/seq_len;indices=origins[:,None]+np.arange(h,dtype=np.int64)[None,:]
        residuals.append((z[indices].astype(np.float64)-mean[:,None,:]).reshape(-1,z.shape[1]));totals.append(len(origins)*h)
        index_contract.append(dict(H=h,origin_first=seq_len,origin_last_inclusive=train_end-h,origins=len(origins),lead_offsets=[0,h-1],history_offsets=[-seq_len,-1],target_first=seq_len,target_last_inclusive=train_end-1,elements_per_channel=len(origins)*h,total_horizon_weight='1/'+str(len(horizons))))
    denominator=math.lcm(*totals);require(len(horizons)*denominator<=np.iinfo(np.int64).max,'Exact rational weights exceed int64 budget')
    weights=np.concatenate([np.full(n,denominator//n,dtype=np.int64) for n in totals]);values=np.concatenate(residuals,axis=0)
    medians=[];mads=[];delta=[]
    for c in range(z.shape[1]):
        median=left_weighted_median(values[:,c],weights);mad=left_weighted_median(np.abs(values[:,c]-median),weights);medians.append(median);mads.append(mad);delta.append(max(1.,2.5*1.4826*mad))
    return dict(delta_float64=delta,delta_float32=np.asarray(delta,dtype=np.float32).astype(float).tolist(),weighted_residual_median=medians,weighted_MAD=mads,train_end=train_end,seq_len=seq_len,horizons=list(horizons),channels=z.shape[1],z32_C_order_sha256=array_sha(z),z32_shape=list(z.shape),z32_dtype='float32',prefix_sum_dtype='float64',median_rule='stable ascending; smallest value with exact integer cumulative weight >= half; ties select left CDF quantile',integer_weight_common_denominator=denominator,integer_total_weight=len(horizons)*denominator,integer_weight_per_observation_by_H={str(h):denominator//n for h,n in zip(horizons,totals)},index_contract=index_contract,threshold_rule='max(1.0, 2.5 * 1.4826 * weightedMAD); no upper cap; each H total weight1/4',residual_rule='z32[o:o+H].float64 - (FP64prefix[o]-FP64prefix[o-L])/L',TRAIN_only=True,seed_independent=True)

def mixed_elements(pred,target,delta):
    """Final standardized B,H,C space; delta fixed; no channel reweighting."""
    import torch
    error=pred.float()-target.float();require(error.ndim==3 and delta.shape==(error.shape[-1],),'B,H,C error and per-channel delta required')
    d=delta.detach().to(device=error.device,dtype=error.dtype).view(1,1,-1);absolute=error.abs();q=torch.minimum(absolute,d);squared=error.square();huber2=q.square()+2*d*(absolute-q)
    return .5*squared+.5*huber2,error,squared

def add_channel_adapter_regularization(loss,model,micro_elements,macro_elements):
    """Element-weight a consumed microbatch mean, including a short macro tail."""
    import torch
    if not 0<micro_elements<=macro_elements:raise ValueError('Invalid micro/macro element count')
    penalty=model.channel_adapter_penalty()
    if not isinstance(penalty,torch.Tensor) or penalty.ndim!=0 or not torch.isfinite(penalty):
        raise FloatingPointError('Channel adapter penalty must be a finite scalar tensor')
    if not penalty.requires_grad:raise RuntimeError('Channel adapter penalty lost autograd')
    beta=float(model.aux_channel_penalty_beta)
    if not math.isfinite(beta) or beta<=0:raise ValueError('Active channel adapter penalty requires beta>0')
    return loss+beta*penalty*(micro_elements/macro_elements),penalty
