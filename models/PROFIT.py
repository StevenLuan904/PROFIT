__all__ = ['PROFIT', 'Model']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch_dct as dct

from layers.PROFIT_backbone import PROFIT_backbone
from layers.RevIN import RevIN


class PROFIT(nn.Module):
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, 
                 d_v: Optional[int] = None, norm: str = 'BatchNorm', attn_dropout: float = 0., 
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None, 
                 attn_mask: Optional[Tensor] = None, res_attention: bool = True, pre_norm: bool = False, 
                 store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True, 
                 pretrain_head: bool = False, head_type: str = 'flatten', verbose: bool = False, **kwargs):
        super().__init__()
        
        # Configuration parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        individual = configs.individual
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        decomposition = configs.decomposition
        
        # PROFIT-specific parameters
        self.decomp_method = getattr(configs, 'decomp_method', 'dct')
        self.k_leaders = getattr(configs, 'patch_leaders', 0)
        self.variable_batch_size = getattr(configs, 'variable_batch_size', 32)
        self.use_leader_integration = getattr(configs, 'use_leader_integration', 1) == 1
        self.corr_threshold = getattr(configs, 'corr_threshold', 0.1)
        self.decomposition = decomposition
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.context_window = context_window
        if not isinstance(self.k_leaders, int) or not 0 <= self.k_leaders < c_in:
            raise ValueError(f'patch_leaders must be an integer in [0, {c_in - 1}], got {self.k_leaders}')
        if not isinstance(decomposition, int) or not 1 <= decomposition <= context_window:
            raise ValueError('decomposition must be an integer between 1 and seq_len')
        if not isinstance(self.variable_batch_size, int) or self.variable_batch_size < 1:
            raise ValueError('variable_batch_size must be a positive integer')
        if self.decomp_method != 'dct':
            raise ValueError(f'Only dct frequency decomposition is implemented, got {self.decomp_method!r}')
        
        # RevIN normalization
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Learnable integration parameters
        self.base_original_weight = nn.Parameter(torch.tensor(0.6))
        self.base_leader_weight = nn.Parameter(torch.tensor(0.4))
        self.correlation_scale = nn.Parameter(torch.tensor(1.0))
        self.leader_adjustments = nn.Parameter(torch.zeros(self.k_leaders))
        
        # Leader state tracking
        self.last_leader_ids = None
        self.last_shift = None
        self.last_r = None
        
        # Model architecture
        self.expansion_factor = self.k_leaders + 1
        c_in_effective = c_in * self.expansion_factor
        
        self.component_models = nn.ModuleList([
            PROFIT_backbone(
                c_in=c_in_effective, context_window=context_window, target_window=target_window, 
                patch_len=patch_len, stride=stride, max_seq_len=max_seq_len,
                n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                dropout=dropout, act=act, key_padding_mask=key_padding_mask, 
                padding_var=padding_var, attn_mask=attn_mask, 
                res_attention=res_attention, pre_norm=pre_norm, 
                store_attn=store_attn, pe=pe, learn_pe=learn_pe, 
                fc_dropout=fc_dropout, head_dropout=head_dropout, 
                padding_patch=padding_patch, pretrain_head=pretrain_head, 
                head_type=head_type, individual=individual, 
                revin=False, affine=affine, subtract_last=subtract_last, 
                verbose=verbose, expansion_factor=self.expansion_factor,
                original_c_in=c_in, **kwargs
            ) for _ in range(self.decomposition)
        ])
        # Non-persistent buffers add no trainable parameters or checkpoint keys.
        split_points = torch.linspace(0, context_window, decomposition + 1, dtype=torch.long)
        frequency_masks = torch.zeros(decomposition, context_window)
        for index in range(decomposition):
            frequency_masks[index, split_points[index]:split_points[index + 1]] = 1
        self.register_buffer('_frequency_masks', frequency_masks, persistent=False)
        self.register_buffer('_context_indices', torch.arange(context_window), persistent=False)
        self.phase_target_batch_size = getattr(configs, 'phase_target_batch_size', 128)
        self.phase_fallback_batch_size = getattr(configs, 'phase_fallback_batch_size', 32)
        self.phase_dense_threshold = getattr(configs, 'phase_dense_threshold', 128)
        self.frequency_band_batch_size = getattr(configs, 'frequency_band_batch_size',
                                                decomposition if c_in <= self.phase_dense_threshold else 1)
        if self.phase_target_batch_size < 1 or self.phase_fallback_batch_size < 1:
            raise ValueError('Phase target/fallback batch sizes must be positive')
        if self.frequency_band_batch_size < 1:
            raise ValueError('frequency_band_batch_size must be positive')
    
    
    def forward(self, x):
        if self.revin:
            x = self.revin_layer(x, 'norm')

        x = x.permute(0, 2, 1)
        x_expanded = self._expand_with_leaders(x)
        outputs = []
        # A single DCT is reused; inverse transforms are evaluated one band at a
        # time so no [B,C,K,L,F] coefficients and IDCT workspaces coexist.
        for cur_f, component_input in enumerate(self._iter_frequency_components(x_expanded)):
            component_output = self.component_models[cur_f](component_input)
            outputs.append(component_output)
        
        x_out = torch.stack(outputs, dim=-1).sum(dim=-1)
        x_out = x_out.permute(0, 2, 1)

        if self.revin:
            x_out = self.revin_layer(x_out, 'denorm')

        return x_out
    
    def _expand_with_leaders(self, x, leader_ids=None, shift=None, r=None, mode="phase"):
        B, C, L = x.shape
        device = x.device
        if self.k_leaders == 0:
            return x.unsqueeze(-1)
        
        if leader_ids is None or shift is None or r is None:
            with torch.no_grad():
                leader_ids, shift, r = self.get_leader_info_phas(x)
        
        x_expanded = x.new_zeros(B, C, L, self.k_leaders + 1)
        x_expanded[:, :, :, 0] = x
        
        base_indices = self._context_indices.view(1, 1, L, 1)
        shift_expanded = shift.unsqueeze(2)
        shifted_indices = torch.clamp(base_indices - shift_expanded, 0, L - 1)
        
        leader_ids_expanded = leader_ids.unsqueeze(2).expand(B, C, L, self.k_leaders)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, C, L, self.k_leaders)
        
        leader_sequences = x[batch_idx, leader_ids_expanded, shifted_indices]
        
        r_expanded = r.unsqueeze(2)
        correlation_mask = (r_expanded < 0).to(dtype=x.dtype)
        leader_sequences = leader_sequences * (1 - 2 * correlation_mask)
        
        r_abs_expanded = r.abs().unsqueeze(2)
        threshold_mask = r_abs_expanded < self.corr_threshold
        original_repeated = x.unsqueeze(-1).expand(-1, -1, -1, self.k_leaders)
        leader_sequences = torch.where(threshold_mask, original_repeated, leader_sequences)

        x_expanded[:, :, :, 1:] = leader_sequences
        return x_expanded

    def phase_correlation(self, x, variable_batch_size=32, eps=1e-8):
        """Streaming top-K with exact PyTorch tie-selection compatibility.

        Unique maxima need only running K candidates. If an exact tie can
        affect top-K order/membership, recompute those rows in small groups and
        invoke torch.topk over the full candidate axis, matching corrected_v1.
        FFTs of the input are calculated once and shared by all chunks/rows.
        """
        B, C, L = x.shape
        device = x.device
        K = self.k_leaders
        if C <= self.phase_dense_threshold:
            return self._phase_correlation_dense(x, variable_batch_size, eps)
        if K == 0:
            empty = torch.empty(B, C, 0, dtype=torch.long, device=device)
            return empty, empty.clone(), x.new_empty(B, C, 0)
        
        rfft = torch.fft.rfft(x, dim=-1)
        norm_rfft = rfft / rfft.abs().clamp(min=eps)
        conjugate = norm_rfft.conj()
        output_ids, output_shifts, output_values = [], [], []
        for target_start in range(0, C, self.phase_target_batch_size):
            target_end = min(C, target_start + self.phase_target_batch_size)
            target_count = target_end - target_start
            target_ids = torch.arange(target_start, target_end, device=device)
            best_values = x.new_empty(B, target_count, 0)
            best_ids = torch.empty(B, target_count, 0, device=device, dtype=torch.long)
            best_shifts = best_ids.clone()
            tied_rows = torch.zeros(B, target_count, device=device, dtype=torch.bool)
            for leader_start in range(0, C, variable_batch_size):
                leader_end = min(C, leader_start + variable_batch_size)
                leader_ids = torch.arange(leader_start, leader_end, device=device)
                cross_phase = (norm_rfft[:, target_start:target_end].unsqueeze(2)
                               * conjugate[:, leader_start:leader_end].unsqueeze(1))
                peak_values, peak_indices = torch.fft.irfft(cross_phase, n=L, dim=-1).max(dim=-1)
                peak_values = peak_values - 2 * (target_ids[:, None] == leader_ids[None, :]).to(peak_values.dtype)
                peak_indices = torch.where(peak_indices > L // 2, peak_indices - L, peak_indices)
                ids = leader_ids.view(1, 1, -1).expand(B, target_count, -1)
                merged_values = torch.cat((best_values, peak_values), dim=-1)
                merged_ids = torch.cat((best_ids, ids), dim=-1)
                merged_shifts = torch.cat((best_shifts, peak_indices), dim=-1)
                count = min(K + 1, merged_values.shape[-1])
                values, positions = merged_values.topk(count, dim=-1)
                if count > 1:
                    tied_rows |= (values[..., :-1] == values[..., 1:]).any(dim=-1)
                keep = min(K, count)
                best_values = values[..., :keep]
                best_ids = merged_ids.gather(-1, positions[..., :keep])
                best_shifts = merged_shifts.gather(-1, positions[..., :keep])
            # topk does not define a stable order for equal values. Repeating
            # its full-axis call only for affected rows preserves that behavior.
            pairs = tied_rows.nonzero(as_tuple=False)
            for row_start in range(0, pairs.shape[0], self.phase_fallback_batch_size):
                pairs_chunk = pairs[row_start:row_start + self.phase_fallback_batch_size]
                batch_ids, local_target_ids = pairs_chunk.unbind(-1)
                global_target_ids = local_target_ids + target_start
                scores, shifts = [], []
                for leader_start in range(0, C, variable_batch_size):
                    leader_end = min(C, leader_start + variable_batch_size)
                    candidate_ids = torch.arange(leader_start, leader_end, device=device)
                    cross_phase = (norm_rfft[batch_ids, global_target_ids].unsqueeze(1)
                                   * conjugate[batch_ids, leader_start:leader_end])
                    values, indices = torch.fft.irfft(cross_phase, n=L, dim=-1).max(dim=-1)
                    values = values - 2 * (global_target_ids[:, None] == candidate_ids[None, :]).to(values.dtype)
                    scores.append(values)
                    shifts.append(torch.where(indices > L // 2, indices - L, indices))
                scores = torch.cat(scores, dim=-1)
                shifts = torch.cat(shifts, dim=-1)
                values, ids = scores.topk(K, dim=-1)
                best_values[batch_ids, local_target_ids] = values
                best_ids[batch_ids, local_target_ids] = ids
                best_shifts[batch_ids, local_target_ids] = shifts.gather(-1, ids)
            output_ids.append(best_ids)
            output_shifts.append(best_shifts)
            output_values.append(best_values)
        return (torch.cat(output_ids, dim=1), torch.cat(output_shifts, dim=1),
                torch.cat(output_values, dim=1))

    def _phase_correlation_dense(self, x, variable_batch_size=32, eps=1e-8):
        """Low-channel backend: avoid streaming merges/synchronization overhead."""
        B, C, L = x.shape
        K = self.k_leaders
        if K == 0:
            empty = torch.empty(B, C, 0, dtype=torch.long, device=x.device)
            return empty, empty.clone(), x.new_empty(B, C, 0)
        spectrum = torch.fft.rfft(x, dim=-1)
        normalized = spectrum / spectrum.abs().clamp(min=eps)
        conjugate = normalized.conj()
        peaks, lags = [], []
        for start in range(0, C, variable_batch_size):
            cross_phase = normalized.unsqueeze(2) * conjugate[:, start:start + variable_batch_size].unsqueeze(1)
            values, indices = torch.fft.irfft(cross_phase, n=L, dim=-1).max(dim=-1)
            peaks.append(values)
            lags.append(torch.where(indices > L // 2, indices - L, indices))
        peaks = torch.cat(peaks, dim=-1)
        lags = torch.cat(lags, dim=-1)
        peaks = peaks - 2 * torch.eye(C, device=x.device, dtype=peaks.dtype).unsqueeze(0)
        scores, ids = peaks.topk(K, dim=-1)
        return ids, lags.gather(-1, ids), scores
    
    def get_leader_info_phas(self, x):
        B, C, L = x.shape
        leader_ids, shift, r = self.phase_correlation(x, variable_batch_size=self.variable_batch_size)
        return leader_ids, shift, r

    def _decompose_frequencies(self, x):
        # Compatibility/analysis API. The training forward streams these bands.
        x = x.permute(0, 1, 3, 2)
        B, C, K, L = x.shape
        
        X = dct.dct(x, norm='ortho')
        F = self.decomposition
        masks = self._frequency_masks.to(dtype=x.dtype).transpose(0, 1).view(1, 1, 1, L, F)

        # torch_dct transforms its last axis: keep the time-frequency axis last.
        masked_coefficients = (X.unsqueeze(-1) * masks).movedim(-1, -2)
        bands = dct.idct(masked_coefficients, norm='ortho').movedim(-2, -1)
        bands = bands.permute(0, 1, 3, 2, 4)
        return bands

    def _iter_frequency_components(self, x):
        spectrum = dct.dct(x.permute(0, 1, 3, 2), norm='ortho')
        masks = self._frequency_masks.to(dtype=x.dtype)
        for start in range(0, self.decomposition, self.frequency_band_batch_size):
            # Optional small groups trade FFT launch overhead for workspace size.
            group = masks[start:start + self.frequency_band_batch_size]
            coefficients = spectrum.unsqueeze(-2) * group.view(1, 1, 1, -1, spectrum.shape[-1])
            components = dct.idct(coefficients, norm='ortho')
            for offset in range(group.shape[0]):
                yield components[..., offset, :].permute(0, 1, 3, 2)
    
    def count_parameters(self):
        """Print the number of learnable parameters in the model."""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total learnable parameters: {total_params:,}')
        return total_params


# Standard forecasting interface alias.
Model = PROFIT
