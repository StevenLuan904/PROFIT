"""Exact quality inference from consecutive observed rows and a frozen TRAIN scaler."""
from __future__ import annotations
import numpy as np
import torch
try:
 from quality_query_execution_20261030_v2 import QualityQueryExecution
except ModuleNotFoundError:
 from quality_query_execution import QualityQueryExecution

class QualityRowExecution:
 """Reuse overlapping history rows with the original model and leader universe.

 Input is a chronological [L+B-1,C] raw observed-row array. Its B output
 windows are [i:i+L] for i=0..B-1; each excludes its own forecast origin.
 mean/scale are the already fitted TRAIN StandardScaler arrays in the exact
 input column order. Model weights, native lag/padding and DCT are unchanged.
 """
 def __init__(self,model,quality_indices,mean,scale):
  if model.training or any(m.training for m in model.modules()):raise ValueError('Use a fully evaluated frozen model')
  if 'forward' in model.__dict__ or 'get_leader_info_phas' in model.__dict__:raise ValueError('Use the native forward and native phase estimator')
  self.model=model;self.length=int(model.context_window);self.device=next(model.parameters()).device;self.mean=np.asarray(mean,dtype=np.float64);self.scale=np.asarray(scale,dtype=np.float64)
  if self.mean.ndim!=1 or self.scale.shape!=self.mean.shape or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or not (self.scale>0).all():raise ValueError('A finite fitted TRAIN scaler is required')
  self.executor=QualityQueryExecution(model,quality_indices);self.active=False
 def __enter__(self):
  if self.active:raise RuntimeError('Execution context already active')
  self.executor.__enter__();self.active=True;return self
 def __exit__(self,*args):
  self.active=False;return self.executor.__exit__(*args)
 @property
 def last_phase(self):return self.executor.last_phase
 def standardize_rows(self,rows):
  rows=np.asarray(rows,dtype=np.float64)
  if rows.ndim!=2 or rows.shape[1]!=len(self.mean) or rows.shape[0]<self.length:raise ValueError('Expected chronological [L+B-1,C] observed rows')
  # Match the original two FP64 pointwise operations before the FP32 cast.
  return ((rows-self.mean)/self.scale).astype(np.float32)
 @torch.inference_mode()
 def predict_rows(self,rows,return_cpu=True):
  if not self.active:raise RuntimeError('Enter the execution context before prediction')
  if torch.is_autocast_enabled():raise ValueError('Use the qualified FP32 inference path')
  normalized=torch.from_numpy(self.standardize_rows(rows)).to(self.device)
  windows=normalized.unfold(0,self.length,1).permute(0,2,1).contiguous()
  prediction=self.executor(windows)
  return prediction.cpu() if return_cpu else prediction
