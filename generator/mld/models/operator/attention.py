from typing import Optional, Callable, Tuple

import torch
import torch.nn as nn

from .utils import get_clone, get_clones, get_activation_fn


class SkipTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, act=None):
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm
        self.act = get_activation_fn(act)
        assert num_layers % 2 == 1

        num_block = (num_layers - 1) // 2
        self.input_blocks = get_clones(encoder_layer, num_block)
        self.middle_block = get_clone(encoder_layer)
        self.output_blocks = get_clones(encoder_layer, num_block)
        self.linear_blocks = get_clones(
            nn.Linear(2 * self.d_model, self.d_model), num_block
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        x = src
        xs = []

        for module in self.input_blocks:
            x = module(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            xs.append(x)

        x = self.middle_block(
            x, src_mask=mask, src_key_padding_mask=src_key_padding_mask
        )

        for module, linear in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            x = self.act(self.norm(x))

        return (x,)


def build_causal_mask(latent_dim, feature_dim, device):
    unit_size = feature_dim // latent_dim
    dim = latent_dim * 2 + feature_dim
    mask = torch.zeros([dim, dim], dtype=torch.bool, device=device)
    latent_mask = torch.triu(
        torch.ones(latent_dim, latent_dim, device=device), diagonal=1
    )
    feature_mask = torch.triu(
        torch.ones(feature_dim, feature_dim, device=device), diagonal=1
    )
    latent_feature_mask = torch.ones([latent_dim, feature_dim], device=device)
    feature_latent_mask = torch.zeros([feature_dim, latent_dim], device=device)
    for i in range(latent_dim):
        latent_feature_mask[i, : (i + 1) * unit_size] = 0
        feature_latent_mask[: i * unit_size, i] = 1
    mask[:latent_dim, :latent_dim] = latent_mask
    mask[:latent_dim, latent_dim : 2 * latent_dim] = latent_mask
    mask[latent_dim : 2 * latent_dim, :latent_dim] = latent_mask
    mask[latent_dim : 2 * latent_dim, latent_dim : 2 * latent_dim] = latent_mask
    mask[:latent_dim, 2 * latent_dim :] = latent_feature_mask
    mask[latent_dim : 2 * latent_dim, 2 * latent_dim :] = latent_feature_mask
    mask[2 * latent_dim :, :latent_dim] = feature_latent_mask
    mask[2 * latent_dim :, latent_dim : 2 * latent_dim] = feature_latent_mask
    mask[2 * latent_dim :, 2 * latent_dim :] = feature_mask
    return mask


class CausalSkipTransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        latent_dim=15,
        feature_dim=60,
        norm=None,
        act=None,
    ):
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm
        self.act = get_activation_fn(act)
        assert num_layers % 2 == 1

        self.latent_dim = latent_dim
        self.feature_dim = feature_dim

        num_block = (num_layers - 1) // 2
        self.input_blocks = get_clones(encoder_layer, num_block)
        self.middle_block = get_clone(encoder_layer)
        self.output_blocks = get_clones(encoder_layer, num_block)
        self.linear_blocks = get_clones(
            nn.Linear(2 * self.d_model, self.d_model), num_block
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src,
        mask=None,
        src_key_padding_mask=None,
        is_prefix=False,
        is_token=False,
    ):
        if not hasattr(self, "attention_mask"):
            self.attention_mask = build_causal_mask(
                self.latent_dim, self.feature_dim, src.device
            )
            self.prefix_attention_mask = build_causal_mask(5, 20, src.device)
            self.token_attention_mask = build_causal_mask(1, 4, src.device)
        x = src
        xs = []
        if is_prefix:
            src_mask = self.prefix_attention_mask
        elif is_token:
            src_mask = self.token_attention_mask
        else:
            src_mask = self.attention_mask

        for module in self.input_blocks:
            x = module(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            xs.append(x)

        x = self.middle_block(
            x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )

        for module, linear in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            x = self.act(self.norm(x))

        return (x,)


class SkipTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, act=None):
        super().__init__()
        self.d_model = decoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm
        self.act = get_activation_fn(act)
        assert num_layers % 2 == 1

        num_block = (num_layers - 1) // 2
        self.input_blocks = get_clones(decoder_layer, num_block)
        self.middle_block = get_clone(decoder_layer)
        self.output_blocks = get_clones(decoder_layer, num_block)
        self.linear_blocks = get_clones(
            nn.Linear(2 * self.d_model, self.d_model), num_block
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt
        xs = []

        for module in self.input_blocks:
            x = module(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            xs.append(x)

        x = self.middle_block(
            x,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        for module, linear in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm:
            x = self.act(self.norm(x))

        return (x,)


def build_self_causal_mask(size, device):
    return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()


def build_cross_causal_mask(tgt_len, mem_len, device):
    mask = torch.ones(tgt_len, mem_len, device=device, dtype=torch.bool)
    for i in range(tgt_len):
        visible_mem_len = int(i / tgt_len * mem_len) + 1
        mask[i, :visible_mem_len] = False
    return mask


class CausalSkipTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, act=None):
        super().__init__()
        self.d_model = decoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm
        self.act = get_activation_fn(act)
        assert num_layers % 2 == 1

        num_block = (num_layers - 1) // 2
        self.input_blocks = get_clones(decoder_layer, num_block)
        self.middle_block = get_clone(decoder_layer)
        self.output_blocks = get_clones(decoder_layer, num_block)
        self.linear_blocks = get_clones(
            nn.Linear(2 * self.d_model, self.d_model), num_block
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        # Build or refresh masks if sizes/devices changed
        tgt_len = tgt.shape[0]
        mem_len = memory.shape[0]
        device = tgt.device
        if (
            not hasattr(self, "self_attention_mask")
            or self.self_attention_mask.shape != (tgt_len, tgt_len)
            or self.self_attention_mask.device != device
        ):
            self.self_attention_mask = build_self_causal_mask(tgt_len, device)
        if (
            not hasattr(self, "cross_attention_mask")
            or self.cross_attention_mask.shape != (tgt_len, mem_len)
            or self.cross_attention_mask.device != device
        ):
            self.cross_attention_mask = build_cross_causal_mask(
                tgt_len, mem_len, device
            )
        tgt_mask = self.self_attention_mask
        memory_mask = self.cross_attention_mask

        x = tgt
        xs = []

        for module in self.input_blocks:
            x = module(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            xs.append(x)

        x = self.middle_block(
            x,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        for module, linear in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm:
            x = self.act(self.norm(x))

        return (x,)


class TransformerEncoder(nn.Module):
    def __init__(
        self, encoder_layer, num_layers, norm=None, act=None, return_intermediate=False
    ):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.norm = norm
        self.act = get_activation_fn(act)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        intermediate = []
        index = 0
        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

            if self.return_intermediate:
                intermediate.append(output)

        if self.norm:
            output = self.act(self.norm(output))

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoder(nn.Module):
    def __init__(
        self, decoder_layer, num_layers, norm=None, act=None, return_intermediate=False
    ):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.norm = norm
        self.act = get_activation_fn(act)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        output = tgt
        intermediate = []
        index = 0
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

            if self.return_intermediate:
                intermediate.append(output)

        if self.norm:
            output = self.act(self.norm(output))

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        norm_eps=1e-5,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.activation_name = activation
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(
            d_model, dim_feedforward if activation != "geglu" else dim_feedforward * 2
        )
        self.activation = (
            get_activation_fn(activation) if activation != "geglu" else nn.GELU()
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(
            src,
            src,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if self.activation_name == "geglu":
            src2, gate = self.linear1(src).chunk(2, dim=-1)
            src2 = src2 * self.activation(gate)
        else:
            src2 = self.activation(self.linear1(src))
        src2 = self.linear2(self.dropout(src2))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2 = self.self_attn(
            src2,
            src2,
            value=src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        if self.activation_name == "geglu":
            src2, gate = self.linear1(src2).chunk(2, dim=-1)
            src2 = src2 * self.activation(gate)
        else:
            src2 = self.activation(self.linear1(src2))
        src2 = self.linear2(self.dropout(src2))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask)
        return self.forward_post(src, src_mask, src_key_padding_mask)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        norm_eps=1e-5,
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.activation_name = activation
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(
            d_model, dim_feedforward if activation != "geglu" else dim_feedforward * 2
        )
        self.activation = (
            get_activation_fn(activation) if activation != "geglu" else nn.GELU()
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        tgt2 = self.self_attn(
            tgt,
            tgt,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=tgt,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if self.activation_name == "geglu":
            tgt2, gate = self.linear1(tgt).chunk(2, dim=-1)
            tgt2 = tgt2 * self.activation(gate)
        else:
            tgt2 = self.activation(self.linear1(tgt))
        tgt2 = self.linear2(self.dropout(tgt2))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(
            tgt2,
            tgt2,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=tgt2,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        if self.activation_name == "geglu":
            tgt2, gate = self.linear1(tgt2).chunk(2, dim=-1)
            tgt2 = tgt2 * self.activation(gate)
        else:
            tgt2 = self.activation(self.linear1(tgt2))
        tgt2 = self.linear2(self.dropout(tgt2))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
        )
