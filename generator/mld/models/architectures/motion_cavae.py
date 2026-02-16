from typing import Optional

import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution

from mld.models.operator.attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    CausalSkipTransformerEncoder,
    CausalSkipTransformerDecoder,
)
from mld.models.operator.position_encoding import build_position_encoding


class MotionCAVAE(nn.Module):
    def __init__(
        self,
        nfeats,
        latent_dim=[1, 256],
        hidden_dim=None,
        force_pre_post_proj=False,
        ff_size=1024,
        num_layers=9,
        num_heads=4,
        dropout=0.1,
        arch="encoder_decoder",
        normalize_before=False,
        norm_eps=1e-5,
        activation="gelu",
        norm_post=True,
        activation_post=None,
        position_embedding="learned",
    ):
        super(MotionCAVAE, self).__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1] if hidden_dim is None else hidden_dim
        add_pre_post_proj = force_pre_post_proj or (
            hidden_dim is not None and hidden_dim != latent_dim[-1]
        )
        self.latent_pre = (
            nn.Linear(self.latent_dim, latent_dim[-1])
            if add_pre_post_proj
            else nn.Identity()
        )
        self.latent_post = (
            nn.Linear(latent_dim[-1], self.latent_dim)
            if add_pre_post_proj
            else nn.Identity()
        )

        self.arch = arch

        self.query_pos_encoder = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding
        )

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
        if self.arch == "encoder_causal_decoder":
            self.encoder = CausalSkipTransformerEncoder(
                encoder_layer, num_layers, 15, 60, encoder_norm, activation_post
            )
        else:
            self.encoder = SkipTransformerEncoder(
                encoder_layer, num_layers, encoder_norm, activation_post
            )

        if self.arch == "all_encoder":
            decoder_norm = (
                nn.LayerNorm(self.latent_dim, eps=norm_eps) if norm_post else None
            )
            self.decoder = SkipTransformerEncoder(
                encoder_layer, num_layers, decoder_norm, activation_post
            )
        else:
            self.query_pos_decoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding
            )

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

            if self.arch == "encoder_decoder":
                decoder_class = SkipTransformerDecoder
            elif self.arch == "encoder_causal_decoder":
                decoder_class = CausalSkipTransformerDecoder
            else:
                raise ValueError(f"Not support architecture: {self.arch}!")

            self.decoder = decoder_class(
                decoder_layer, num_layers, decoder_norm, activation_post
            )

        self.global_motion_token = nn.Parameter(
            torch.randn(self.latent_size * 2, self.latent_dim)
        )
        self.skel_embedding = nn.Linear(nfeats, self.latent_dim)

        self.final_layer = nn.Linear(self.latent_dim, nfeats)

    def forward(self, features, mask):
        z, dist = self.encode(features, mask)
        feats_rst = self.decode(z, mask)
        return feats_rst, z, dist

    @torch.no_grad()
    def encode_token(self, features):
        assert (
            self.arch == "encoder_causal_decoder"
        ), "Only designed for autoregressive diffusion with causal attention"
        if not hasattr(self, "query_pos_encoder_token"):
            pe = self.query_pos_encoder.pe
            term1 = pe[:1]
            term2 = pe[self.latent_size : self.latent_size + 1]
            term3 = pe[2 * self.latent_size : 2 * self.latent_size + 4]
            self.query_pos_encoder_token = torch.cat([term1, term2, term3])
            term1 = self.global_motion_token[:1]
            term2 = self.global_motion_token[self.latent_size : self.latent_size + 1]
            prefix_token = torch.cat([term1, term2])
            self.global_motion_token_token = prefix_token

        bs, nframes, nfeats = features.shape
        x = self.skel_embedding(features)
        x = x.permute(1, 0, 2)
        dist = torch.tile(self.global_motion_token_token[:, None, :], (1, bs, 1))
        xseq = torch.cat([dist, x])
        xseq = xseq + self.query_pos_encoder_token

        dist = self.encoder(xseq, is_token=True)[0][: dist.shape[0]]
        dist = self.latent_pre(dist)

        mu = dist[0:1]
        logvar = dist[1:]

        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()

        latent = latent.permute(1, 0, 2)
        return latent, dist

    @torch.no_grad()
    def encode_prefix(self, features, mask):
        assert (
            self.arch == "encoder_causal_decoder"
        ), "Only designed for autoregressive diffusion with causal attention"
        if not hasattr(self, "query_pos_encoder_prefix"):
            pe = self.query_pos_encoder.pe
            term1 = pe[:5]
            term2 = pe[self.latent_size : self.latent_size + 5]
            term3 = pe[2 * self.latent_size : 2 * self.latent_size + 20]
            self.query_pos_encoder_prefix = torch.cat([term1, term2, term3])
            term1 = self.global_motion_token[:5]
            term2 = self.global_motion_token[self.latent_size : self.latent_size + 5]
            prefix_token = torch.cat([term1, term2])
            self.global_motion_token_prefix = prefix_token

        bs, nframes, nfeats = features.shape
        x = self.skel_embedding(features)
        x = x.permute(1, 0, 2)
        dist = torch.tile(self.global_motion_token_prefix[:, None, :], (1, bs, 1))
        dist_masks = torch.ones((bs, dist.shape[0]), dtype=torch.bool, device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)
        xseq = torch.cat([dist, x])
        xseq = xseq + self.query_pos_encoder_prefix

        dist = self.encoder(xseq, src_key_padding_mask=~aug_mask, is_prefix=True)[0][
            : dist.shape[0]
        ]
        dist = self.latent_pre(dist)

        mu = dist[0:5]
        logvar = dist[5:]

        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()

        latent = latent.permute(1, 0, 2)
        return latent, dist

    def encode(self, features, mask):
        bs, nframes, nfeats = features.shape
        x = self.skel_embedding(features)
        x = x.permute(1, 0, 2)
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))
        dist_masks = torch.ones((bs, dist.shape[0]), dtype=torch.bool, device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)
        xseq = torch.cat((dist, x), 0)

        xseq = self.query_pos_encoder(xseq)
        dist = self.encoder(xseq, src_key_padding_mask=~aug_mask)[0][: dist.shape[0]]
        dist = self.latent_pre(dist)

        mu = dist[0 : self.latent_size, ...]
        logvar = dist[self.latent_size :, ...]

        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        latent = latent.permute(1, 0, 2)
        return latent, dist

    def decode(self, z, mask, features=None, use_gt_ar=None):
        z = self.latent_post(z)
        z = z.permute(1, 0, 2)
        features = features.permute(1, 0, 2) if features is not None else None
        if self.arch == "all_encoder":
            bs, nframes = mask.shape
            queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
            xseq = torch.cat((z, queries), axis=0)
            z_mask = torch.ones(
                (bs, self.latent_size), dtype=torch.bool, device=z.device
            )
            aug_mask = torch.cat((z_mask, mask), axis=1)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(xseq, src_key_padding_mask=~aug_mask)[0][z.shape[0] :]
            output = self.final_layer(output)
        elif self.arch in ["encoder_decoder", "encoder_causal_decoder"]:
            bs, nframes = mask.shape
            queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
            queries = self.query_pos_decoder(queries)
            output = self.decoder(tgt=queries, memory=z, tgt_key_padding_mask=~mask)[0]
            output = self.final_layer(output)
        else:
            raise ValueError(f"Not support architecture: {self.arch}!")

        output[~mask.T] = 0
        feats = output.permute(1, 0, 2)
        return feats
