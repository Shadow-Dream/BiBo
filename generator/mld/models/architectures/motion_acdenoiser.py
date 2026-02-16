from typing import Optional, Union

import torch
import torch.nn as nn

from mld.models.operator.embeddings import TimestepEmbedding, Timesteps
from mld.models.operator.attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from mld.models.operator.utils import get_clones, get_activation_fn, zero_module
from mld.models.operator.position_encoding import build_position_encoding


class MotionACDenoiser(nn.Module):
    def __init__(
        self,
        latent_dim=[1, 256],
        hidden_dim=None,
        text_dim=768,
        time_dim=768,
        ff_size=1024,
        num_layers=9,
        num_heads=4,
        dropout=0.1,
        normalize_before=False,
        norm_eps=1e-5,
        activation="gelu",
        norm_post=True,
        activation_post=None,
        flip_sin_to_cos=True,
        freq_shift=0,
        time_act_fn="silu",
        time_post_act_fn=None,
        position_embedding="learned",
        arch="trans_enc",
        add_mem_pos=True,
        force_pre_post_proj=False,
        text_act_fn="relu",
        time_cond_proj_dim=None,
        zero_init_cond=True,
        prefix_latent_len=5,
        use_length_emb=False,
    ):
        super(MotionACDenoiser, self).__init__()

        self.latent_dim = latent_dim[-1] if hidden_dim is None else hidden_dim
        self.prefix_latent_len = prefix_latent_len
        add_pre_post_proj = force_pre_post_proj or (
            hidden_dim is not None and hidden_dim != latent_dim[-1]
        )
        self.latent_pre = (
            nn.Linear(latent_dim[-1], self.latent_dim)
            if add_pre_post_proj
            else nn.Identity()
        )
        self.latent_post = (
            nn.Linear(self.latent_dim, latent_dim[-1])
            if add_pre_post_proj
            else nn.Identity()
        )

        self.prefix_proj = (
            nn.Linear(latent_dim[-1], self.latent_dim)
            if add_pre_post_proj
            else nn.Identity()
        )

        self.use_length_emb = use_length_emb
        if self.use_length_emb:
            self.cond_proj = (
                nn.Linear(46, self.latent_dim) if add_pre_post_proj else nn.Identity()
            )
        else:
            self.cond_proj = (
                nn.Linear(36, self.latent_dim) if add_pre_post_proj else nn.Identity()
            )

        self.arch = arch
        self.time_cond_proj_dim = time_cond_proj_dim

        self.time_proj = Timesteps(time_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(
            time_dim,
            self.latent_dim,
            time_act_fn,
            post_act_fn=time_post_act_fn,
            cond_proj_dim=time_cond_proj_dim,
            zero_init_cond=zero_init_cond,
        )
        self.emb_proj = nn.Sequential(
            get_activation_fn(text_act_fn), nn.Linear(text_dim, self.latent_dim)
        )

        self.query_pos = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding
        )
        if self.arch == "trans_enc":

            encoder_layer = TransformerEncoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
                norm_eps,
            )

            encoder_norm = (
                nn.LayerNorm(self.latent_dim, eps=norm_eps) if norm_post else None
            )
            self.encoder = SkipTransformerEncoder(
                encoder_layer, num_layers, encoder_norm, activation_post
            )

        elif self.arch == "trans_dec":
            if add_mem_pos:
                self.mem_pos = build_position_encoding(
                    self.latent_dim, position_embedding=position_embedding
                )
            else:
                self.mem_pos = None

            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
                norm_eps,
            )

            decoder_norm = (
                nn.LayerNorm(self.latent_dim, eps=norm_eps) if norm_post else None
            )
            self.decoder = SkipTransformerDecoder(
                decoder_layer, num_layers, decoder_norm, activation_post
            )
        else:
            raise ValueError(f"Not supported architecture: {self.arch}!")

    def forward(
        self, sample, timestep, prefix, encoder_hidden_states, cond, cond_mask, lengths
    ):
        # 1. dimension matching (pre)
        sample = sample.permute(1, 0, 2)
        sample = self.latent_pre(sample)

        prefix = prefix.permute(1, 0, 2)
        prefix = self.prefix_proj(prefix)

        # 2. time_embedding
        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        time_emb = self.time_embedding(time_emb).unsqueeze(0)

        # 3. text embedding projection
        encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
        text_emb_latent = self.emb_proj(encoder_hidden_states)

        if lengths is None:
            cond = cond.reshape(-1, 27)
            cond_mask = cond_mask.reshape(-1, 9)
            cond = torch.cat([cond, cond_mask], dim=-1)
            cond = self.cond_proj(cond)
            length_mask = None
        elif self.use_length_emb:
            length_mask = None
            cond = cond.reshape(-1, 27)
            cond_mask = cond_mask.reshape(-1, 9)
            lengths = torch.eye(10, device=cond.device)[lengths]
            cond = torch.cat([cond, cond_mask, lengths], dim=-1)
            cond = self.cond_proj(cond)
        else:
            cond = cond.reshape(-1, 27)
            cond_mask = cond_mask.reshape(-1, 9)
            cond = torch.cat([cond, cond_mask], dim=-1)
            cond = self.cond_proj(cond)

            cond_length = 8
            length_mask = torch.zeros(
                [lengths.shape[0], 10 + cond_length, 10 + cond_length],
                dtype=torch.bool,
                device=cond.device,
            )
            length_mask[:, :-1, -1] = True
            length_mask[torch.arange(lengths.shape[0]), lengths, -1] = False
            B, L, _ = length_mask.shape
            length_mask = length_mask.unsqueeze(1).expand(B, 4, L, L)
            length_mask = length_mask.reshape(B * 4, L, L)
        cond = cond.unsqueeze(0)

        # 4. text + time embedding + prefix
        emb_latent = torch.cat((time_emb, text_emb_latent, prefix, cond), 0)

        # 5. transformer
        if self.arch == "trans_enc":
            xseq = torch.cat((sample, emb_latent), axis=0)
            xseq = self.query_pos(xseq)
            tokens = self.encoder(xseq, mask=length_mask)[0]
        elif self.arch == "trans_dec":
            sample = self.query_pos(sample)
            if self.mem_pos:
                emb_latent = self.mem_pos(emb_latent)
            tokens = self.decoder(sample, emb_latent)[0]
        else:
            raise TypeError(f"{self.arch} is not supported")

        if self.arch == "trans_enc":
            sample = tokens[: sample.shape[0]]
        elif self.arch == "trans_dec":
            sample = tokens
        else:
            raise TypeError(f"{self.arch} is not supported")

        # 6. dimension matching (post)
        sample = self.latent_post(sample)
        sample = sample.permute(1, 0, 2)
        return sample
