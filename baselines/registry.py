"""Construct the bundled PROFIT core with the canonical parameter mapping."""
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = dict(seq_len=360, pred_len=12, e_layers=2, n_heads=8, d_model=64,
                    d_ff=256, dropout=0.05, fc_dropout=0.05, head_dropout=0.0,
                    individual=False, patch_len=16, stride=8, padding_patch='end',
                    revin=1, affine=0, subtract_last=0, decomposition=5,
                    patch_leaders=3, variable_batch_size=32, decomp_method='dct',
                    kernel_size=25, in_dim=1, leader_num=4, state_num=8,
                    temperature=1.0, local_rank=-1, seg_len=24, win_size=2,
                    num_routers=10)



def build_model(config, channels, device):
    source=Path(config.get('model_source_root',ROOT)).resolve()
    for module_name in ('models.PAFNet','models.PROFITAux','layers.PAFNet_backbone'):
        loaded=sys.modules.get(module_name)
        if loaded is not None and getattr(loaded,'__file__',None):
            try: Path(loaded.__file__).resolve().relative_to(source)
            except ValueError: raise RuntimeError('A different model source is already imported; use one independent process per source')
    sys.path.insert(0,str(source))
    hp=dict(DEFAULT_MODEL)
    hp.update(config.get('model_params',{}))
    hp.update(enc_in=channels,c_out=channels,dec_in=channels,
              seq_len=config['seq_len'],pred_len=config['horizon'])
    args=SimpleNamespace(**hp)
    name=config['model']
    provenance={'model':name,'source_root':str(source),'resolved_model_params':hp}
    if name in ('PROFIT','PROFIT_original','PAFNet'):
        module=importlib.import_module('models.PAFNet')
        model=module.Model(args)
    elif name=='PROFIT_Aux':
        model=importlib.import_module('models.PROFITAux').Model(args)
        provenance['auxiliary_changes']={k:v for k,v in hp.items() if k.startswith('aux_')}
        provenance['core_contract']='phase/lag and DCT-band mechanisms unchanged; independent normalization/head only'
    else:
        raise ValueError('This isolated release supports PROFIT_Aux, PROFIT, PROFIT_original and PAFNet only')
    components=getattr(model,'component_models',None)
    if components is not None:
        import torch.nn as nn
        encoders=[component.backbone for component in components]
        bn_count=sum(isinstance(module,nn.modules.batchnorm._BatchNorm) for encoder in encoders for module in encoder.modules())
        ln_count=sum(isinstance(module,nn.LayerNorm) for encoder in encoders for module in encoder.modules())
        declared=getattr(model,'encoder_normalization_counts',None)
        if declared is not None and (declared['BatchNorm']!=bn_count or declared['LayerNorm']!=ln_count):
            raise ValueError('Wrapper encoder normalization counts disagree with independent registry count')
        provenance['encoder_norm_counts']={'BatchNorm':bn_count,'LayerNorm':ln_count,'scope':'component.backbone modules; component.head excluded'}
        provenance['auxiliary_head_layernorm_count']=sum(isinstance(module,nn.LayerNorm) for component in components for module in component.head.modules())
        requested=hp.get('encoder_norm')
        if requested=='LayerNorm' and bn_count:
            raise ValueError('Requested encoder LayerNorm but actual backbone contains BatchNorm')
        if requested=='BatchNorm' and not bn_count:
            raise ValueError('Requested encoder BatchNorm but no backbone BatchNorm modules were found')
    provenance['class_module']=model.__class__.__module__
    provenance['parameters']=sum(p.numel() for p in model.parameters())
    provenance['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model.to(device), provenance
