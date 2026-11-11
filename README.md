# PROFIT

Core PyTorch implementation of PROFIT for multivariate time-series forecasting.
The model selects and aligns phase-correlated histories, decomposes them into DCT
frequency bands, and combines patch representations within each band.

`models/PROFIT.py` contains the core architecture; `models/PROFITAux.py` adds
the residual readout and centered low-rank channel adapter. Both expose a
`PROFIT` class; `Model` remains available as an alias.

## Usage

Install dependencies with `python -m pip install -r requirements.txt`, then run
from the repository root. This example shows the forward interface:

```python
from types import SimpleNamespace
import torch
from models.PROFITAux import PROFIT

cfg = SimpleNamespace(
    enc_in=11, seq_len=360, pred_len=24,
    e_layers=1, n_heads=4, d_model=32, d_ff=128,
    dropout=0.1, fc_dropout=0.1, head_dropout=0.0,
    individual=False, patch_len=32, stride=16, padding_patch="end",
    revin=1, affine=1, subtract_last=0,
    decomposition=3, decomp_method="dct", patch_leaders=1,
    variable_batch_size=32, use_leader_integration=1, corr_threshold=0.1,
    aux_norm="core", aux_head="summary_residual",
    aux_head_hidden=16, aux_head_dropout=0.1, aux_channel_rank=4,
)
model = PROFIT(cfg).eval()
with torch.no_grad():
    prediction = model(torch.randn(2, 360, 11))
print(prediction.shape)  # torch.Size([2, 24, 11])
```

Inputs and outputs use `[batch, time, variables]`. Example settings illustrate
the interface. Training drivers, datasets, checkpoints, and plotting code are
not included. Source hashes are recorded in `PROVENANCE.json`; the inherited
license is retained in `LICENSE`.
