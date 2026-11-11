"""Optional normalization/readout variants around the frozen PROFIT core.

No phase/leader/boundary/frequency/encoder/cross-attention operation is changed.
The default core normalization and linear head instantiate exactly the core.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from models.PROFIT import PROFIT as CoreModel


class ShrinkRevIN(nn.Module):
    """Invertible normalization with learned shrinkage toward scaler prior 0/1.

    Window moments are detached, following RevIN. The shrinkage gate remains
    differentiable. Statistics are retained only for the matching denorm call.
    """

    def __init__(self, channels: int, init: float = 0.9, affine: bool = False,
                 eps: float = 1e-5, scope: str = 'scalar', learnable: bool = True):
        super().__init__()
        if not 0.0 <= init <= 1.0 or (learnable and not 0.0 < init < 1.0):
            raise ValueError('Learnable shrinkage requires 0 < init < 1; fixed shrinkage allows endpoints')
        if scope not in ('scalar', 'channel'):
            raise ValueError('aux_shrink_scope must be scalar or channel')
        if eps <= 0:
            raise ValueError('aux_norm_eps must be positive')
        self.channels = channels
        self.eps = float(eps)
        self.affine = bool(affine)
        self.scope = scope
        self.learnable = bool(learnable)
        shape = (channels,) if scope == 'channel' else (1,)
        if self.learnable:
            self.shrink_logit = nn.Parameter(torch.full(shape, math.log(init / (1 - init))))
        else:
            self.register_buffer('fixed_shrinkage', torch.full(shape, init))
        if self.affine:
            self.affine_log_scale = nn.Parameter(torch.zeros(channels))
            self.affine_bias = nn.Parameter(torch.zeros(channels))
        self._center = None
        self._scale = None

    @property
    def shrinkage(self):
        return torch.sigmoid(self.shrink_logit) if self.learnable else self.fixed_shrinkage

    def forward(self, x, mode):
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError('ShrinkRevIN expects [batch, time, channels]')
        if mode == 'norm':
            mean = x.mean(dim=1, keepdim=True).detach()
            variance = x.var(dim=1, keepdim=True, unbiased=False).detach()
            gate = self.shrinkage.view(1, 1, -1)
            self._center = gate * mean
            self._scale = torch.sqrt((1 - gate) + gate * (variance + self.eps))
            output = (x - self._center) / self._scale
            if self.affine:
                output = output * self.affine_log_scale.exp() + self.affine_bias
            return output
        if mode == 'denorm':
            if self._center is None or self._scale is None:
                raise RuntimeError('norm must precede matching denorm')
            if x.shape[0] != self._center.shape[0]:
                raise ValueError('denorm batch does not match saved norm statistics')
            output = x
            if self.affine:
                output = (output - self.affine_bias) * (-self.affine_log_scale).exp()
            return output * self._scale + self._center
        raise ValueError(f'Unsupported normalization mode: {mode!r}')


class SharedBottleneckHead(nn.Module):
    """Per-channel readout, shared across channels within one frequency band."""

    def __init__(self, features: int, hidden: int, horizon: int, dropout: float):
        super().__init__()
        if hidden < 1 or not 0 <= dropout < 1:
            raise ValueError('Require aux_head_hidden >= 1 and 0 <= aux_head_dropout < 1')
        self.flatten = nn.Flatten(start_dim=-2)
        self.readout = nn.Sequential(nn.Linear(features, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, horizon))

    def forward(self, x):
        return self.readout(self.flatten(x))


class ResidualReadout(nn.Module):
    """Zero-initialized correction to the untouched per-band original readout."""

    def __init__(self, original, features, d_model, hidden, horizon, dropout, summary):
        super().__init__()
        if hidden < 1 or not 0 <= dropout < 1:
            raise ValueError('Invalid residual-head hidden size or dropout')
        self.original = original
        self.summary = bool(summary)
        input_size = 2 * d_model if self.summary else features
        layers = [nn.LayerNorm(input_size, elementwise_affine=False)] if self.summary else []
        layers += [nn.Linear(input_size, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, horizon)]
        self.correction = nn.Sequential(*layers)
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, x):
        summary = torch.cat((x.mean(dim=-1), x[..., -1]), dim=-1) if self.summary else x.flatten(start_dim=-2)
        return self.original(x) + self.correction(summary)


class CenteredLinearChannelAdapter(nn.Module):
    """Bias-free linear adapter with shared orthogonal down-projection.

    Only the up factor is channel-specific. Centering that factor removes its
    common mapping component; it does not force the mean prediction to zero.
    """

    def __init__(self, features, channels, rank, horizon):
        super().__init__()
        self.down = nn.Linear(features, rank, bias=False)
        self.up = nn.Parameter(torch.zeros(channels, horizon, rank))
        self.scale = 1.0 / math.sqrt(rank)
        nn.init.orthogonal_(self.down.weight)

    def forward(self, encoded):
        latent = self.down(encoded)
        centered_up = self.up - self.up.mean(dim=0, keepdim=True)
        return torch.einsum('bcr,chr->bch', latent, centered_up) * self.scale


class CenteredLinearChannelReadout(nn.Module):
    """Add an adapter to the unchanged original linear/summary readout."""

    def __init__(self, original, features, channels, rank, horizon, collect_penalty):
        super().__init__()
        self.original = original
        self.adapter = CenteredLinearChannelAdapter(features, channels, rank, horizon)
        self.collect_penalty = bool(collect_penalty)
        self._penalty_delta = None

    def train(self, mode=True):
        self._penalty_delta = None
        return super().train(mode)

    def forward(self, x):
        self._penalty_delta = None
        prediction = self.original(x)
        # Reuse the exact existing Flatten_Head flattening and axes. It is
        # stateless; the original shared head/summary forward remains intact.
        linear_head = self.original.original if isinstance(self.original, ResidualReadout) else self.original
        encoded = linear_head.flatten(x)
        delta = self.adapter(encoded)
        if self.collect_penalty and self.training and torch.is_grad_enabled():
            self._penalty_delta = delta
        return prediction + delta


class PROFIT(CoreModel):
    """Frozen-core model with explicitly opted-in auxiliary substitutions."""

    def __init__(self, configs, *args, **kwargs):
        norm = getattr(configs, 'aux_norm', 'core')
        head = getattr(configs, 'aux_head', 'linear')
        channel_rank = getattr(configs, 'aux_channel_rank', 0)
        penalty_beta = float(getattr(configs, 'aux_channel_penalty_beta', 0.0))
        if type(channel_rank) is not int or channel_rank < 0:
            raise ValueError('aux_channel_rank must be a nonnegative integer')
        if not math.isfinite(penalty_beta) or penalty_beta < 0:
            raise ValueError('aux_channel_penalty_beta must be finite and nonnegative')
        if channel_rank and configs.individual:
            raise ValueError('aux_channel_rank requires individual=False')
        if channel_rank and head not in ('linear', 'summary_residual'):
            raise ValueError('aux_channel_rank supports the shared linear or summary_residual readout')
        if norm not in ('core', 'identity', 'shrink'):
            raise ValueError('aux_norm must be core, identity, or shrink')
        if head not in ('linear', 'mlp', 'summary_residual', 'flat_residual'):
            raise ValueError('Unknown auxiliary readout')
        core_config = copy.copy(configs)
        if norm == 'identity':
            core_config.revin = 0
        elif norm == 'shrink':
            core_config.revin = 1
        encoder_norm = getattr(configs, 'encoder_norm', None)
        if encoder_norm is not None:
            if encoder_norm not in ('BatchNorm', 'LayerNorm'):
                raise ValueError('encoder_norm must be BatchNorm or LayerNorm')
            kwargs['norm'] = encoder_norm
        super().__init__(core_config, *args, **kwargs)
        self.aux_norm = norm
        self.aux_head = head
        self.aux_channel_rank = channel_rank
        self.aux_channel_penalty_beta = penalty_beta
        self.channel_adapter_penalty_enabled = channel_rank > 0 and penalty_beta > 0
        self.encoder_norm = encoder_norm or kwargs.get('norm', 'BatchNorm')
        encoder_modules = [module for component in self.component_models
                           for module in component.backbone.modules()]
        self.encoder_normalization_counts = dict(
            BatchNorm=sum(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in encoder_modules),
            LayerNorm=sum(isinstance(module, nn.LayerNorm) for module in encoder_modules))
        # The frozen core accepts norm but does not pass it into TSTiEncoder.
        # v1 supports the proven BatchNorm path; reject misleading declarations.
        if self.encoder_norm == 'LayerNorm' and self.encoder_normalization_counts['BatchNorm']:
            raise ValueError('Requested LayerNorm was not instantiated by the frozen core; use a separately verified auxiliary normalization revision')
        if norm == 'shrink':
            self.revin_layer = ShrinkRevIN(configs.enc_in,
                init=float(getattr(configs, 'aux_shrink_init', 0.9)),
                affine=bool(getattr(configs, 'aux_norm_affine', False)),
                eps=float(getattr(configs, 'aux_norm_eps', 1e-5)),
                scope=getattr(configs, 'aux_shrink_scope', 'scalar'),
                learnable=bool(getattr(configs, 'aux_shrink_learnable', True)))
        if head == 'mlp':
            if configs.individual:
                raise ValueError('Aux shared MLP head requires individual=False')
            for component in self.component_models:
                component.head = SharedBottleneckHead(component.head_nf,
                    hidden=int(getattr(configs, 'aux_head_hidden', 64)),
                    horizon=configs.pred_len,
                    dropout=float(getattr(configs, 'aux_head_dropout', 0.0)))
        elif head in ('summary_residual', 'flat_residual'):
            for component in self.component_models:
                summary = head == 'summary_residual'
                component.head = ResidualReadout(component.head, component.head_nf, component.d_model,
                    hidden=int(getattr(configs, 'aux_head_hidden', 16 if summary else 8)),
                    horizon=configs.pred_len,
                    dropout=float(getattr(configs, 'aux_head_dropout', 0.1)), summary=summary)
        head_init = getattr(configs, 'aux_head_init', getattr(configs, 'head_init', 'default'))
        if head_init not in ('default', 'small', 'zero'):
            raise ValueError('aux_head_init must be default, small, or zero')
        if head_init != 'default':
            std = float(getattr(configs, 'aux_head_init_std', 1e-3))
            if std <= 0:
                raise ValueError('aux_head_init_std must be positive')
            for component in self.component_models:
                if isinstance(component.head, ResidualReadout):
                    final_layers = [component.head.correction[-1]]
                elif isinstance(component.head, SharedBottleneckHead):
                    final_layers = [component.head.readout[-1]]
                elif component.head.individual:
                    final_layers = list(component.head.linears)
                else:
                    final_layers = [component.head.linear]
                for final in final_layers:
                    if head_init == 'zero':
                        nn.init.zeros_(final.weight)
                    else:
                        nn.init.normal_(final.weight, std=std)
                    if final.bias is not None:
                        nn.init.zeros_(final.bias)
        # All original core/head construction and head_init finish first. One
        # CPU fork protects the external RNG start while allowing distinct
        # adapter matrices across bands. Registry moves the complete model to
        # its target device only after this CPU construction.
        if channel_rank:
            with torch.random.fork_rng(devices=[]):
                for component in self.component_models:
                    component.head = CenteredLinearChannelReadout(component.head,
                        component.head_nf, configs.enc_in, channel_rank, configs.pred_len,
                        collect_penalty=self.channel_adapter_penalty_enabled)

    def channel_adapter_named_parameters(self):
        """Actual factor parameters for the independent optimizer WD group."""
        if not self.aux_channel_rank:
            return []
        return [(f'component_models.{index}.head.adapter.{name}', parameter)
                for index, component in enumerate(self.component_models)
                for name, parameter in component.head.adapter.named_parameters()]

    def channel_adapter_penalty(self):
        """Consume raw mean squared adapter output in standardized units.

        Beta is applied by the harness. No StandardScaler inverse transform is
        involved. Both the summed band residual and actual RevIN inverse gain
        keep their gradients; caches are cleared before returning the graph.
        """
        if not self.channel_adapter_penalty_enabled:
            raise RuntimeError('Channel adapter penalty is disabled')
        heads = [component.head for component in self.component_models]
        deltas = [head._penalty_delta for head in heads]
        for head in heads:
            head._penalty_delta = None
        if not self.training or not torch.is_grad_enabled() or any(delta is None for delta in deltas):
            raise RuntimeError('Penalty requires one matching training forward with gradients enabled')
        normalized_delta = torch.stack(deltas, dim=0).sum(dim=0).float()  # B,C,H
        if self.revin:
            batch, channels, _ = normalized_delta.shape
            zero = normalized_delta.new_zeros(batch, 1, channels)
            gain = (self.revin_layer(torch.ones_like(zero), 'denorm') -
                    self.revin_layer(zero, 'denorm')).transpose(1, 2)  # B,C,1
            standardized_delta = normalized_delta * gain.float()
        else:
            standardized_delta = normalized_delta
        return standardized_delta.square().mean()


# Standard forecasting interface alias.
Model = PROFIT
