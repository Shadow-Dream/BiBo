import time
import inspect
import logging
from typing import Optional

import tqdm
import numpy as np
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler

from mld.data.base import BaseDataModule
from mld.config import instantiate_from_config
from mld.utils.temos_utils import lengths_to_mask, remove_padding
from mld.utils.utils import (
    count_parameters,
    get_guidance_scale_embedding,
    extract_into_tensor,
    control_loss_calculate,
)
from mld.data.humanml.utils.plot_script import plot_3d_motion
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel

logger = logging.getLogger(__name__)


class WeightedSumMasked(torch.nn.Module):
    """
    Learn global per-item weights and compute a mask-normalized weighted average over items.
    Keeps output scale invariant to the number of active conditions, supports empty-set (returns ~0).
    """

    def __init__(self, n_items):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(n_items))

    def forward(self, E, mask):
        # E: [B, J, D], mask: [B, J] (bool)
        w = torch.softmax(self.logits, dim=0)  # [J]
        w = w.unsqueeze(0).expand_as(mask).float()  # [B, J]
        w = w * mask.float()
        denom = w.sum(1, keepdim=True).clamp_min(1e-6)  # avoid div0 for empty-set
        w = w / denom
        return (E * w[..., None]).sum(1)


class EmbedTargetLocMultiFast(nn.Module):

    def __init__(self, joint_names, latent_dim, hidden_dim=None):
        super().__init__()
        self.joint_names = list(joint_names)
        self.J = len(self.joint_names)
        H = hidden_dim or latent_dim

        # One parameter set per condition: w1[J,H,3], b1[J,H], w2[J,D,H], b2[J,D]
        self.w1 = nn.Parameter(torch.empty(self.J, H, 3))
        self.b1 = nn.Parameter(torch.zeros(self.J, H))
        self.w2 = nn.Parameter(torch.empty(self.J, latent_dim, H))
        self.b2 = nn.Parameter(torch.zeros(self.J, latent_dim))

        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

        self.agg = WeightedSumMasked(self.J)

    def forward(self, x, mask):
        # x: [B, J, 3], mask: [B, J] (bool)
        h = torch.einsum("bji,jhi->bjh", x, self.w1) + self.b1  # [B,J,H]
        h = F.silu(h)
        e = torch.einsum("bjh,jdh->bjd", h, self.w2) + self.b2  # [B,J,D]
        e = e * mask[..., None].float()
        out = self.agg(e, mask)  # [B,D]
        return out


class CondFusion(torch.nn.Module):
    """
    Embed per-joint 3D conditions into a single vector and fuse into context (e.g., text embeddings).
    - Per-joint MLP: 3 -> H -> Df
    - Masked, normalized aggregation over joints
    - Projection to context dim and optional LayerNorm + learnable scale
    """

    def __init__(self, joint_names, fusion_dim=256, hidden_dim=None, scale_init=1.0):
        super().__init__()
        self.joint_names = list(joint_names)
        self.J = len(self.joint_names)
        H = hidden_dim or fusion_dim
        Df = fusion_dim

        # ParameterDict of small MLPs
        self.target_loc_emb = torch.nn.ParameterDict(
            {
                n: torch.nn.Sequential(
                    torch.nn.Linear(3, H), torch.nn.SiLU(), torch.nn.Linear(H, Df)
                )
                for n in self.joint_names
            }
        )
        self.idx = {n: i for i, n in enumerate(self.joint_names)}
        self.agg = WeightedSumMasked(self.J)

        # Lazy projection to match encoder_hidden_states last dim
        self.proj = None
        self.ln = None
        self.scale = torch.nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, cond_3d, cond_mask, ctx):
        """cond_3d: [B, J, 3]; cond_mask: [B, J]; ctx: [B, D] or [B, L, D]"""
        B, J, _ = cond_3d.shape
        D_ctx = ctx.shape[-1]

        # Per-joint embeddings [B, J, Df]
        E = []
        for n in self.joint_names:
            j = self.idx[n]
            e = self.target_loc_emb[n](cond_3d[:, j, :])  # [B, Df]
            E.append(e.unsqueeze(1))
        E = torch.cat(E, dim=1)  # [B, J, Df]
        Em = self.agg(E, cond_mask)  # [B, Df]

        # Lazy modules
        if self.proj is None:
            self.proj = torch.nn.Linear(Em.shape[-1], D_ctx, bias=True).to(
                Em.device, Em.dtype
            )
        if self.ln is None:
            self.ln = torch.nn.LayerNorm(D_ctx).to(Em.device)

        add = self.ln(self.proj(Em)) * self.scale  # [B, D_ctx]

        if ctx.dim() == 3:
            return ctx + add[:, None, :]
        else:
            return ctx + add


class CACLDM(BaseModel):
    def __init__(self, cfg, datamodule):
        super().__init__()

        self.cfg = cfg
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.fixed_len = cfg.DATASET.FIXED_LEN
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.prefix_latent_len = cfg.model.prefix_latent_len
        self.latent_unit = self.fixed_len // self.latent_dim[0]
        self.prefix_len = self.latent_unit * self.prefix_latent_len
        self.datamodule = datamodule
        self.control_mode = (
            cfg.model.control_mode if hasattr(cfg.model, "control_mode") else "last"
        )
        self.joint_names = [
            "pelvis",
            "left_hip",
            "right_hip",
            "spine1",
            "left_knee",
            "right_knee",
            "spine2",
            "left_ankle",
            "right_ankle",
            "spine3",
            "left_foot",
            "right_foot",
            "neck",
            "left_collar",
            "right_collar",
            "head",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
        ]

        if cfg.model.guidance_scale == "dynamic":
            s_cfg = cfg.model.scheduler
            self.guidance_scale = s_cfg.cfg_step_map[s_cfg.num_inference_steps]
            logger.info(f"Guidance Scale set as {self.guidance_scale}")

        self.text_encoder = instantiate_from_config(cfg.model.text_encoder)
        self.vae = instantiate_from_config(cfg.model.motion_vae)
        self.denoiser = instantiate_from_config(cfg.model.denoiser)
        # --- Cond fusion (MDM-like) ---
        # self.use_cond_fusion =  True
        self.cond_names = [
            "traj",
            "heading",
            "vel",
            "pelvis",
            "left_foot",
            "right_foot",
            "head",
            "left_wrist",
            "right_wrist",
        ]
        # fusion_dim = cfg.model.get('cond_fusion_dim', self.latent_dim[1])
        # fusion_scale = cfg.model.get('cond_fusion_scale', 1.0)
        # self.cond_fusion = CondFusion(self.cond_names, fusion_dim=fusion_dim, scale_init=fusion_scale)

        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.alphas = torch.sqrt(self.scheduler.alphas_cumprod)
        self.sigmas = torch.sqrt(1 - self.scheduler.alphas_cumprod)

        # Cond loss training options
        self.cond_loss_weight = cfg.TRAIN.get("cond_loss_weight", 1.0)
        self.cond_loss_warmup_steps = cfg.TRAIN.get("cond_loss_warmup_steps", 0)
        self.cond_mask_curriculum_steps = cfg.TRAIN.get("cond_mask_curriculum_steps", 0)
        self.cond_mask_curriculum_set = cfg.TRAIN.get(
            "cond_mask_curriculum_set", [["traj", "heading"]]
        )
        self.cond_loss_t_weight = cfg.TRAIN.get(
            "cond_loss_t_weight", "none"
        )  # 'none', 'alpha_bar', 'sqrt_alpha_bar', 'snr'
        self.vel_normalize_mode = cfg.TRAIN.get(
            "vel_normalize_mode", "position_stats"
        )  # 'position_stats', 'zero_mean'

        # Training step counter (will be set by training loop)
        self.training_step = 0

        self._get_t2m_evaluator(cfg)

        self.metric_list = cfg.METRIC.TYPE
        self.configure_metrics()

        self.feats2joints = datamodule.feats2joints

        self.vae_scale_factor = cfg.model.get("vae_scale_factor", 1.0)
        self.guidance_uncondp = cfg.model.get("guidance_uncondp", 0.0)

        logger.info(f"vae_scale_factor: {self.vae_scale_factor}")
        logger.info(f"prediction_type: {self.scheduler.config.prediction_type}")
        logger.info(f"guidance_scale: {self.guidance_scale}")
        logger.info(f"guidance_uncondp: {self.guidance_uncondp}")

        self.raw_mean = np.load("data/Mean_raw.npy")
        self.raw_std = np.load("data/Std_raw.npy")

        self.dno = (
            instantiate_from_config(cfg.model["noise_optimizer"])
            if cfg.model.get("noise_optimizer")
            else None
        )

        self.summarize_parameters()
        cond_names = [
            "traj",
            "heading",
            "vel",
            "pelvis",
            "left_foot",
            "right_foot",
            "head",
            "left_wrist",
            "right_wrist",
        ]
        self.cond_names = cond_names
        # Single-position control pairs (one position at a time)
        cond_pairs = [
            [],  # No control
            ["head"],
            ["left_wrist"],
            ["right_wrist"],
            ["left_foot"],
            ["right_foot"],
            ["pelvis"],
        ]
        # self.cond_embed = EmbedTargetLocMultiFast(
        #     joint_names=self.cond_names,
        #     latent_dim=self.latent_dim[1]  # Latent channel count used in reverse diffusion.
        # )
        cond_vectors = []
        for cond_pair in cond_pairs:
            cond_vector = torch.zeros(len(cond_names), dtype=bool)
            for cond_name in cond_pair:
                cond_vector[cond_names.index(cond_name)] = True
            cond_vectors.append(cond_vector)
        self.cond_vectors = torch.stack(cond_vectors)

        # Global control metrics tracking
        self.control_metrics = {
            "hand_mae_left": [],
            "hand_mae_right": [],
            "foot_mae_left": [],
            "foot_mae_right": [],
            "head_mae": [],
            "pelvis_mae": [],
            "overall_control_error": [],
            # Real distance metrics (in cm)
            "hand_dist_cm_left": [],
            "hand_dist_cm_right": [],
            "foot_dist_cm_left": [],
            "foot_dist_cm_right": [],
            "head_dist_cm": [],
            "pelvis_dist_cm": [],
            "overall_dist_cm": [],
            "total_samples": 0,
        }

    @property
    def do_classifier_free_guidance(self):
        return self.guidance_scale > 1 and self.denoiser.time_cond_proj_dim is None

    def summarize_parameters(self):
        logger.info(f"VAE Encoder: {count_parameters(self.vae.encoder)}M")
        logger.info(f"VAE Decoder: {count_parameters(self.vae.decoder)}M")
        logger.info(f"Denoiser: {count_parameters(self.denoiser)}M")

    def forward(self, batch):
        texts = batch["text"]
        feats_ref = batch.get("motion")
        lengths = batch["length"]

        if self.do_classifier_free_guidance:
            texts = texts + [""] * len(texts)

        text_emb = self.text_encoder(texts)

        latents = torch.randn((len(lengths), *self.latent_dim), device=text_emb.device)
        mask = batch.get("mask", lengths_to_mask(lengths, text_emb.device))

        latents = self._diffusion_reverse(latents, text_emb)
        feats_rst = self.vae.decode(latents / self.vae_scale_factor, mask)

        joints = self.feats2joints(feats_rst.detach().cpu())
        joints = remove_padding(joints, lengths)

        joints_ref = None
        if feats_ref is not None:
            joints_ref = self.feats2joints(feats_ref.detach().cpu())
            joints_ref = remove_padding(joints_ref, lengths)

        return joints, joints_ref

    def predicted_origin(self, model_output, timesteps, sample):
        self.alphas = self.alphas.to(model_output.device)
        self.sigmas = self.sigmas.to(model_output.device)
        alphas = extract_into_tensor(self.alphas, timesteps, sample.shape)
        sigmas = extract_into_tensor(self.sigmas, timesteps, sample.shape)

        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - sigmas * model_output) / alphas
            pred_epsilon = model_output
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alphas * model_output) / sigmas
        else:
            raise ValueError(
                f"Invalid prediction_type {self.scheduler.config.prediction_type}."
            )

        return pred_original_sample, pred_epsilon

    def _diffusion_reverse(
        self, latents, prefix, encoder_hidden_states, cond, cond_mask, lengths
    ):
        if self.control_mode == "last":
            lengths = None
        else:
            lengths = torch.tensor(lengths).to(latents.device).long()

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        prefix = torch.cat([prefix] * 2) if self.do_classifier_free_guidance else prefix
        cond = torch.cat([cond] * 2) if self.do_classifier_free_guidance else cond
        cond_mask = (
            torch.cat([cond_mask] * 2)
            if self.do_classifier_free_guidance
            else cond_mask
        )
        if lengths is not None:
            lengths = (
                torch.cat([lengths] * 2)
                if self.do_classifier_free_guidance
                else lengths
            )
        # encoder_hidden_states = self.cond_fusion(cond, cond_mask, encoder_hidden_states)
        for i, t in tqdm.tqdm(enumerate(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            model_output = self.denoiser(
                sample=latent_model_input,
                prefix=prefix,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                cond=cond,
                cond_mask=cond_mask,
                lengths=lengths,
            )

            # perform guidance
            if self.do_classifier_free_guidance:
                model_output_text, model_output_uncond = model_output.chunk(2)
                model_output = model_output_uncond + self.guidance_scale * (
                    model_output_text - model_output_uncond
                )

            latents = self.scheduler.step(
                model_output, t, latents, **extra_step_kwargs
            ).prev_sample

        return latents

    def _diffusion_process(
        self, latents, encoder_hidden_states, cond, cond_mask, lengths
    ):

        prefix = latents[:, : self.prefix_latent_len, :]
        latents = latents[:, self.prefix_latent_len :, :]
        if lengths is not None:
            lengths = lengths // 4 - 5 - 1

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
            dtype=torch.long,
        )

        noisy_latents = self.scheduler.add_noise(latents.clone(), noise, timesteps)

        # encoder_hidden_states = self.cond_fusion(cond, cond_mask, encoder_hidden_states)

        model_output = self.denoiser(
            sample=noisy_latents,
            prefix=prefix,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            cond=cond,
            cond_mask=cond_mask,
            lengths=lengths,
        )

        latents_pred, noise_pred = self.predicted_origin(
            model_output, timesteps, noisy_latents
        )

        n_set = {
            "noise": noise,
            "noise_pred": noise_pred,
            "sample_pred": latents_pred,
            "sample_gt": latents,
            "prefix": prefix,
            "timesteps": timesteps,
        }
        return n_set

    def compute_cond(self, feats_ref, lengths=None):
        if (
            not isinstance(self.raw_mean, torch.Tensor)
            or self.raw_mean.device != feats_ref.device
        ):
            self.raw_mean = torch.tensor(self.raw_mean).to(feats_ref.device)
            self.raw_std = torch.tensor(self.raw_std).to(feats_ref.device)

        feats_ref = feats_ref[:, 20:]
        joints_ref = self.feats2joints(feats_ref)
        if lengths is None:
            vel = joints_ref[:, -1, 0] - joints_ref[:, -2, 0]
            joints_ref = joints_ref[:, -1]
        else:
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths, device=feats_ref.device)
            lengths = lengths - 20
            # Use advanced indexing for variable lengths
            batch_size = joints_ref.shape[0]
            vel = []
            joints_final = []
            for i in range(batch_size):
                idx = lengths[i].item()
                vel.append(joints_ref[i, idx, 0] - joints_ref[i, idx - 1, 0])
                joints_final.append(joints_ref[i, idx])
            vel = torch.stack(vel)
            joints_ref = torch.stack(joints_final)

        face_joint_idx = [2, 1, 17, 16]
        r_hip, l_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints_ref[:, r_hip] - joints_ref[:, l_hip]
        across2 = joints_ref[:, sdr_r] - joints_ref[:, sdr_l]
        across = across1 + across2
        across = torch.nn.functional.normalize(across, dim=1)
        up_vector = torch.tensor([0, 1, 0], dtype=across.dtype, device=across.device)
        up_vector = up_vector.unsqueeze(0).expand(
            across.shape[0], -1
        )
        forward = torch.cross(up_vector, across, dim=1)
        forward = torch.nn.functional.normalize(forward, dim=1)
        heading_angle = torch.atan2(forward[:, 0], forward[:, 2])
        heading = torch.stack(
            [
                torch.sin(heading_angle),
                torch.cos(heading_angle),
                torch.zeros_like(heading_angle),
            ],
            dim=1,
        )

        joints_ref = (joints_ref - self.raw_mean) / self.raw_std

        # Velocity normalization based on config
        if self.vel_normalize_mode == "zero_mean":
            vel = vel / (self.raw_std[0] + 1e-8)  # Zero-mean assumption
        else:  # 'position_stats' (default)
            vel = (vel - self.raw_mean[0]) / self.raw_std[0]
        pelvis = joints_ref[:, 0]
        left_foot = joints_ref[:, 10]
        right_foot = joints_ref[:, 11]
        head = joints_ref[:, 15]
        left_wrist = joints_ref[:, 20]
        right_wrist = joints_ref[:, 21]
        traj = pelvis + 0
        vel[:, 1] = 0
        traj[:, 1] = 0

        cond = torch.stack(
            [
                traj,
                heading,
                vel,
                pelvis,
                left_foot,
                right_foot,
                head,
                left_wrist,
                right_wrist,
            ],
            dim=1,
        )
        return cond

    def compute_control_metrics(self, joints_pred, joints_ref, cond_mask, lengths=None):

        # Joint indices mapping
        joint_indices = {
            "pelvis": 0,
            "head": 15,
            "left_wrist": 20,
            "right_wrist": 21,
            "left_foot": 10,
            "right_foot": 11,
        }

        # Condition names mapping to joint indices
        cond_to_joint = {
            "pelvis": 0,
            "head": 15,
            "left_wrist": 20,
            "right_wrist": 21,
            "left_foot": 10,
            "right_foot": 11,
        }

        # Cond mask indices
        cond_indices = {
            "pelvis": 3,
            "left_foot": 4,
            "right_foot": 5,
            "head": 6,
            "left_wrist": 7,
            "right_wrist": 8,
        }

        metrics = {}

        # Handle sequence lengths
        if lengths is not None:
            # Apply length mask to joints
            max_len = joints_pred.shape[1]
            length_mask = (
                torch.arange(max_len, device=joints_pred.device)[None, :]
                < lengths[:, None]
            )
            length_mask = length_mask[
                :, :, None
            ]  # [B, T, 1] for broadcasting with [B, T, 3]
        else:
            length_mask = torch.ones(
                joints_pred.shape[:2] + (1,), device=joints_pred.device
            )  # [B, T, 1]

        for cond_name, joint_idx in cond_to_joint.items():
            cond_idx = cond_indices[cond_name]

            # Check if this condition is active for any sample in the batch
            active_mask = cond_mask[:, cond_idx]  # [B]

            if active_mask.any():
                # Get joints for this condition
                pred_joint = joints_pred[:, :, joint_idx]  # [B, T, 3]
                ref_joint = joints_ref[:, :, joint_idx]  # [B, T, 3]

                # Apply masks
                diff = torch.abs(
                    pred_joint - ref_joint
                )  # [B, T, 3] - Use MAE instead of MSE
                diff = (
                    diff * length_mask
                )  # Apply length mask, broadcasting [B, T, 1] to [B, T, 3]

                # Compute MAE for active samples only
                active_samples = active_mask.nonzero().flatten()
                if len(active_samples) > 0:
                    if lengths is not None:
                        # Weight by actual sequence length for active samples
                        active_lengths = lengths[active_samples].float()
                        mae_per_sample = diff[active_samples].sum(dim=[1, 2]) / (
                            active_lengths * 3
                        )
                    else:
                        mae_per_sample = diff[active_samples].mean(dim=[1, 2])

                    avg_mae = mae_per_sample.mean().item()

                    # Map to metric names
                    if cond_name == "left_wrist":
                        metrics["hand_mae_left"] = avg_mae
                    elif cond_name == "right_wrist":
                        metrics["hand_mae_right"] = avg_mae
                    elif cond_name == "left_foot":
                        metrics["foot_mae_left"] = avg_mae
                    elif cond_name == "right_foot":
                        metrics["foot_mae_right"] = avg_mae
                    elif cond_name == "head":
                        metrics["head_mae"] = avg_mae
                    elif cond_name == "pelvis":
                        metrics["pelvis_mae"] = avg_mae

        # Compute overall control error across all active conditions
        all_active_conds = cond_mask[:, 3:].any(dim=1)  # Skip traj, heading, vel
        if all_active_conds.any():
            overall_errors = []
            for b in range(joints_pred.shape[0]):
                if all_active_conds[b]:
                    active_cond_indices = (
                        cond_mask[b, 3:].nonzero().flatten() + 3
                    )  # Add offset for traj,heading,vel
                    sample_errors = []

                    for cond_idx in active_cond_indices:
                        cond_name = list(cond_indices.keys())[
                            list(cond_indices.values()).index(cond_idx)
                        ]
                        joint_idx = cond_to_joint[cond_name]

                        pred_joint = joints_pred[b, :, joint_idx]  # [T, 3]
                        ref_joint = joints_ref[b, :, joint_idx]  # [T, 3]

                        if lengths is not None:
                            seq_len = lengths[b]
                            error = (
                                torch.abs(pred_joint[:seq_len] - ref_joint[:seq_len])
                                .mean()
                                .item()
                            )
                        else:
                            error = torch.abs(pred_joint - ref_joint).mean().item()

                        sample_errors.append(error)

                    if sample_errors:
                        overall_errors.append(np.mean(sample_errors))

            if overall_errors:
                metrics["overall_control_error"] = np.mean(overall_errors)

        # === Add Real Distance Metrics (in cm) ===
        # Convert joints from normalized space to real coordinates for distance calculation
        if hasattr(self, "raw_mean") and hasattr(self, "raw_std"):
            # Ensure raw_mean and raw_std are tensors on the correct device
            if not isinstance(self.raw_mean, torch.Tensor):
                self.raw_mean = torch.tensor(self.raw_mean, device=joints_pred.device)
                self.raw_std = torch.tensor(self.raw_std, device=joints_pred.device)
            elif self.raw_mean.device != joints_pred.device:
                self.raw_mean = self.raw_mean.to(joints_pred.device)
                self.raw_std = self.raw_std.to(joints_pred.device)

            # Convert to real coordinates (denormalize)
            joints_pred_real = joints_pred * self.raw_std + self.raw_mean
            joints_ref_real = joints_ref * self.raw_std + self.raw_mean

            for cond_name, joint_idx in cond_to_joint.items():
                cond_idx = cond_indices[cond_name]
                active_mask = cond_mask[:, cond_idx]  # [B]

                if active_mask.any():
                    # Get joints for this condition
                    pred_joint_real = joints_pred_real[:, :, joint_idx]  # [B, T, 3]
                    ref_joint_real = joints_ref_real[:, :, joint_idx]  # [B, T, 3]

                    # Calculate 3D Euclidean distance
                    diff_3d = torch.norm(
                        pred_joint_real - ref_joint_real, dim=-1
                    )  # [B, T]
                    diff_3d = diff_3d * length_mask.squeeze(
                        -1
                    )  # Apply length mask [B, T]

                    # Convert to centimeters (assuming raw data is in meters)
                    diff_cm = diff_3d * 100

                    # Compute distance for active samples only
                    active_samples = active_mask.nonzero().flatten()
                    if len(active_samples) > 0:
                        if lengths is not None:
                            # Weight by actual sequence length for active samples
                            active_lengths = lengths[active_samples].float()
                            dist_per_sample = (
                                diff_cm[active_samples].sum(dim=1) / active_lengths
                            )
                        else:
                            dist_per_sample = diff_cm[active_samples].mean(dim=1)

                        avg_dist_cm = dist_per_sample.mean().item()

                        # Map to distance metric names
                        if cond_name == "left_wrist":
                            metrics["hand_dist_cm_left"] = avg_dist_cm
                        elif cond_name == "right_wrist":
                            metrics["hand_dist_cm_right"] = avg_dist_cm
                        elif cond_name == "left_foot":
                            metrics["foot_dist_cm_left"] = avg_dist_cm
                        elif cond_name == "right_foot":
                            metrics["foot_dist_cm_right"] = avg_dist_cm
                        elif cond_name == "head":
                            metrics["head_dist_cm"] = avg_dist_cm
                        elif cond_name == "pelvis":
                            metrics["pelvis_dist_cm"] = avg_dist_cm

            # Compute overall distance error across all active conditions
            if all_active_conds.any():
                overall_dist_errors = []
                for b in range(joints_pred_real.shape[0]):
                    if all_active_conds[b]:
                        active_cond_indices = cond_mask[b, 3:].nonzero().flatten() + 3
                        sample_dist_errors = []

                        for cond_idx in active_cond_indices:
                            cond_name = list(cond_indices.keys())[
                                list(cond_indices.values()).index(cond_idx)
                            ]
                            joint_idx = cond_to_joint[cond_name]

                            pred_joint_real = joints_pred_real[
                                b, :, joint_idx
                            ]  # [T, 3]
                            ref_joint_real = joints_ref_real[b, :, joint_idx]  # [T, 3]

                            if lengths is not None:
                                seq_len = lengths[b]
                                dist_error = (
                                    torch.norm(
                                        pred_joint_real[:seq_len]
                                        - ref_joint_real[:seq_len],
                                        dim=-1,
                                    )
                                    .mean()
                                    .item()
                                )
                            else:
                                dist_error = (
                                    torch.norm(pred_joint_real - ref_joint_real, dim=-1)
                                    .mean()
                                    .item()
                                )

                            # Convert to cm
                            sample_dist_errors.append(dist_error * 100)

                        if sample_dist_errors:
                            overall_dist_errors.append(np.mean(sample_dist_errors))

                if overall_dist_errors:
                    metrics["overall_dist_cm"] = np.mean(overall_dist_errors)

        return metrics

    def update_control_metrics(self, metrics_dict):
        """Update global control metrics with new batch results."""
        for key, value in metrics_dict.items():
            if key in self.control_metrics and key != "total_samples":
                self.control_metrics[key].append(value)
        self.control_metrics["total_samples"] += 1

    def get_control_metrics_summary(self):
        """Get summary statistics of accumulated control metrics."""
        summary = {}
        for key, values in self.control_metrics.items():
            if key != "total_samples" and values:
                summary[f"{key}_mean"] = np.mean(values)
                summary[f"{key}_std"] = np.std(values)
                summary[f"{key}_count"] = len(values)
        summary["total_batches"] = self.control_metrics["total_samples"]
        return summary

    def reset_control_metrics(self):
        """Reset all control metrics."""
        for key in self.control_metrics:
            if key != "total_samples":
                self.control_metrics[key] = []
            else:
                self.control_metrics[key] = 0

    def log_final_control_metrics(self):
        """Log final control metrics summary."""
        if hasattr(self, "control_metrics"):
            summary = self.get_control_metrics_summary()
            logger.info("=== Final Control Metrics Summary ===")
            logger.info(f"Total batches processed: {summary.get('total_batches', 0)}")

            # Log hand metrics
            if "hand_mae_left_mean" in summary:
                logger.info(
                    f"Hand MAE (Left): {summary['hand_mae_left_mean']:.6f} ± {summary.get('hand_mae_left_std', 0):.6f} (n={summary.get('hand_mae_left_count', 0)})"
                )
            if "hand_mae_right_mean" in summary:
                logger.info(
                    f"Hand MAE (Right): {summary['hand_mae_right_mean']:.6f} ± {summary.get('hand_mae_right_std', 0):.6f} (n={summary.get('hand_mae_right_count', 0)})"
                )

            # Log foot metrics
            if "foot_mae_left_mean" in summary:
                logger.info(
                    f"Foot MAE (Left): {summary['foot_mae_left_mean']:.6f} ± {summary.get('foot_mae_left_std', 0):.6f} (n={summary.get('foot_mae_left_count', 0)})"
                )
            if "foot_mae_right_mean" in summary:
                logger.info(
                    f"Foot MAE (Right): {summary['foot_mae_right_mean']:.6f} ± {summary.get('foot_mae_right_std', 0):.6f} (n={summary.get('foot_mae_right_count', 0)})"
                )

            # Log other metrics
            if "head_mae_mean" in summary:
                logger.info(
                    f"Head MAE: {summary['head_mae_mean']:.6f} ± {summary.get('head_mae_std', 0):.6f} (n={summary.get('head_mae_count', 0)})"
                )
            if "pelvis_mae_mean" in summary:
                logger.info(
                    f"Pelvis MAE: {summary['pelvis_mae_mean']:.6f} ± {summary.get('pelvis_mae_std', 0):.6f} (n={summary.get('pelvis_mae_count', 0)})"
                )
            if "overall_control_error_mean" in summary:
                logger.info(
                    f"Overall Control Error: {summary['overall_control_error_mean']:.6f} ± {summary.get('overall_control_error_std', 0):.6f} (n={summary.get('overall_control_error_count', 0)})"
                )

            logger.info("=====================================")
            return summary
        return {}

    def sample_cond(self, feats_ref, lengths=None):
        if self.cond_vectors.device != feats_ref.device:
            self.cond_vectors = self.cond_vectors.to(feats_ref.device)
        cond = self.compute_cond(feats_ref, lengths)

        # Curriculum learning for cond_mask
        if self.training_step < self.cond_mask_curriculum_steps:
            # Use curriculum set during early training
            curriculum_vectors = []
            for cond_pair in self.cond_mask_curriculum_set:
                cond_vector = torch.zeros(len(self.cond_names), dtype=bool)
                for cond_name in cond_pair:
                    if cond_name in self.cond_names:
                        cond_vector[self.cond_names.index(cond_name)] = True
                curriculum_vectors.append(cond_vector)
            curriculum_vectors = torch.stack(curriculum_vectors).to(feats_ref.device)
            cond_mask_indices = torch.randint(
                curriculum_vectors.shape[0], [cond.shape[0]], device=feats_ref.device
            )
            cond_mask = curriculum_vectors[cond_mask_indices]
        else:
            # Use full cond_vectors after curriculum phase
            cond_mask_indices = torch.randint(
                self.cond_vectors.shape[0], [cond.shape[0]], device=feats_ref.device
            )
            cond_mask = self.cond_vectors[cond_mask_indices]

        cond = cond * cond_mask[..., None]
        return cond, cond_mask

    def sample_length(self, lengths):
        lengths = lengths // 4 - 10
        cutoff = torch.rand(lengths.shape[0], device=lengths.device)
        cutoff = cutoff**0.5
        lengths = torch.round(cutoff * lengths).long()
        lengths = (lengths + 10) * 4
        return lengths

    def train_diffusion_forward(self, batch):
        feats_ref = batch["motion"]
        mask = batch["mask"]

        # Prepare lengths (per-frame) according to control mode
        if self.control_mode == "last":
            mask = torch.ones_like(mask)
            lengths_eff = None
        else:
            lengths_all = torch.tensor(batch["length"]).to(mask.device).long()
            # If using long prefix, training should operate on the tail after 120 frames

            lengths_eff = lengths_all
            lengths_eff = self.sample_length(lengths_eff)

        cond, cond_mask = self.sample_cond(feats_ref, lengths_eff)

        with torch.no_grad():
            z, _ = self.vae.encode(feats_ref, mask)
        z = z * self.vae_scale_factor

        # Text guidance (CFG during training)
        text = batch["text"]
        text = ["" if np.random.rand(1) < self.guidance_uncondp else i for i in text]
        text_emb = self.text_encoder(text)

        n_set = self._diffusion_process(z, text_emb, cond, cond_mask, lengths_eff)
        timesteps = n_set["timesteps"]

        loss_dict = dict()
        # Diffusion loss
        if self.scheduler.config.prediction_type == "epsilon":
            model_pred, target = n_set["noise_pred"], n_set["noise"]
        elif self.scheduler.config.prediction_type == "sample":
            model_pred, target = n_set["sample_pred"], n_set["sample_gt"]
        else:
            raise ValueError(
                f"Invalid prediction_type {self.scheduler.config.prediction_type}."
            )
        diff_loss = F.mse_loss(model_pred, target, reduction="mean")
        loss_dict["diff_loss"] = diff_loss

        # Control loss on decoded samples
        sample_pred = n_set["sample_pred"]
        prefix_lat = n_set["prefix"]
        sample_pred = torch.cat([prefix_lat, sample_pred], dim=1)
        z_dec = sample_pred / self.vae_scale_factor
        feats_rst = self.vae.decode(z_dec, mask)
        cond_pred = self.compute_cond(feats_rst, lengths_eff)
        cond_pred = cond_pred * cond_mask[..., None]

        cond_active = cond_mask[..., None].float()  # [B, 9, 1]
        per_elem = F.smooth_l1_loss(
            cond_pred * cond_active, cond * cond_active, reduction="none"
        )
        num_active_per_sample = (
            cond_active.sum(dim=[1, 2]) * cond_pred.size(-1)
        ).clamp_min(1.0)
        cond_loss_per_sample = per_elem.sum(dim=[1, 2]) / num_active_per_sample

        if self.cond_loss_t_weight != "none":
            t_weight = self._get_timestep_weight(timesteps)
            cond_loss = (cond_loss_per_sample * t_weight).mean()
        else:
            cond_loss = cond_loss_per_sample.mean()

        cond_loss_weight = self._get_cond_loss_weight()
        cond_loss = cond_loss * cond_loss_weight
        loss_dict["cond_loss"] = cond_loss

        total_loss = sum(loss_dict.values())
        loss_dict["loss"] = total_loss
        return loss_dict

    def t2m_eval(self, batch):
        texts = batch["text"]
        full_feats_ref = batch["motion"]
        full_mask = batch["mask"]
        full_len = full_feats_ref.shape[1]
        fixed_len = self.fixed_len
        latent_unit = self.latent_unit
        prefix_len = self.prefix_len
        complete_len = fixed_len - prefix_len
        complete_latent_len = complete_len // latent_unit
        lengths = batch["length"]
        word_embs = batch["word_embs"]
        pos_ohot = batch["pos_ohot"]
        text_lengths = batch["text_len"]

        batch_size = len(lengths)
        # Use the last frame as control condition
        cond, cond_mask = self.sample_cond(full_feats_ref, lengths)
        if self.do_classifier_free_guidance:
            texts = texts + [""] * len(texts)
        text_emb = self.text_encoder(texts)

        iterations = (full_len - prefix_len - 1) // complete_len + 1

        prefix = full_feats_ref[:, :prefix_len]
        prefix_mask = full_mask[:, :prefix_len]

        full_feats_rst = []
        for i in range(iterations):
            prefix_mask = torch.ones(
                [batch_size, prefix_len], dtype=torch.bool, device="cuda"
            )
            mask = torch.ones([batch_size, fixed_len], dtype=torch.bool, device="cuda")

            prefix_emb, _ = self.vae.encode_prefix(prefix, prefix_mask)
            prefix_emb = prefix_emb * self.vae_scale_factor
            start_time = time.time()
            latents = torch.randn(
                (mask.shape[0], complete_latent_len, self.latent_dim[1]),
                device=text_emb.device,
            )
            latents = self._diffusion_reverse(
                latents, prefix_emb, text_emb, cond, cond_mask, lengths
            )

            latents = torch.cat([prefix_emb, latents], dim=1)
            latents = latents / self.vae_scale_factor
            feats_rst = self.vae.decode(latents, mask)
            end_time = time.time()
            full_feats_rst.append(feats_rst[:, prefix_len:])

            prefix = feats_rst[:, -prefix_len:]

            full_feats_rst = torch.cat(full_feats_rst, dim=1)
            feats_ref = full_feats_ref[:, :full_len]
            feats_rst = full_feats_rst[:, :full_len]

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(feats_ref)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        feats_ref = self.datamodule.renorm4t2m(feats_ref)

        m_lens = torch.tensor(lengths, device=feats_ref.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        feats_ref = feats_ref[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(
            m_lens,
            eval(f"self.cfg.DATASET.{self.cfg.DATASET.NAME.upper()}.UNIT_LEN"),
            rounding_mode="floor",
        )

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(feats_ref[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[align_idx]

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
        }

        # Compute and update control metrics if control conditions are present
        if hasattr(self, "control_metrics"):
            # Use zeros for no control (as set in this eval mode)
            control_metrics = self.compute_control_metrics(
                joints_rst,
                joints_ref,
                cond_mask,
                m_lens if not self.long_prefix else m_lens,
            )
            if control_metrics:
                self.update_control_metrics(control_metrics)
                rs_set["control_metrics"] = control_metrics

        return rs_set

    def t2m_test(self, batch):
        texts = batch["text"]
        feats_ref = batch["motion"]
        mask = batch["mask"]
        lengths = batch["length"]

        if self.control_mode == "last":
            lengths = None

        cond, cond_mask = self.sample_cond(feats_ref, lengths)
        if self.do_classifier_free_guidance:
            texts = texts + [""] * len(texts)
        text_emb = self.text_encoder(texts)

        prefix = feats_ref[:, :20]
        prefix_mask = mask[:, :20]

        prefix_emb, _ = self.vae.encode_prefix(prefix, prefix_mask)
        prefix_emb = prefix_emb * self.vae_scale_factor

        latents = torch.randn(
            (mask.shape[0], 10, self.latent_dim[1]), device=text_emb.device
        )
        latents = self._diffusion_reverse(
            latents, prefix_emb, text_emb, cond, cond_mask, lengths
        )

        latents = torch.cat([prefix_emb, latents], dim=1) / self.vae_scale_factor
        feats_rst = self.vae.decode(latents, mask)

        # Convert to joints for control metrics
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(feats_ref)

        loss_dict = {}

        cond_pred = self.compute_cond(feats_rst, lengths)
        cond_pred = cond_pred * cond_mask[..., None]
        cond_loss = control_loss_calculate(
            "mask", "l1_smooth", cond_pred, cond, cond_mask[..., None]
        )

        loss_dict["cond_loss"] = cond_loss

        # Compute detailed control metrics
        if hasattr(self, "control_metrics"):
            control_metrics = self.compute_control_metrics(
                joints_rst, joints_ref, cond_mask, lengths
            )
            if control_metrics:
                self.update_control_metrics(control_metrics)
                # Add individual metrics to loss_dict for logging
                for metric_name, metric_value in control_metrics.items():
                    loss_dict[f"test_{metric_name}"] = torch.tensor(metric_value)

        return loss_dict

    def allsplit_step(self, split, batch):
        if split in ["val"]:
            rs_set = self.t2m_eval(batch)

            if self.datamodule.is_mm:
                metric_list = ["MMMetrics"]
            else:
                metric_list = self.metric_list

            for metric in metric_list:
                if metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        rs_set["lat_t"],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"],
                    )
                elif metric == "MMMetrics" and self.datamodule.is_mm:
                    getattr(self, metric).update(
                        rs_set["lat_rm"].unsqueeze(0), batch["length"]
                    )
                else:
                    raise TypeError(f"Not support this metric: {metric}.")

            # Log control metrics summary periodically
            if (
                hasattr(self, "control_metrics")
                and self.control_metrics["total_samples"] % 10 == 0
            ):
                control_summary = self.get_control_metrics_summary()
                logger.info(
                    f"Control Metrics Summary (after {control_summary['total_batches']} batches):"
                )
                for key, value in control_summary.items():
                    if "mean" in key:
                        logger.info(f"  {key}: {value:.6f}")

            return None

        if split in ["test"]:
            loss_dict = self.t2m_test(batch)

            # Log test control metrics
            if hasattr(self, "control_metrics"):
                test_metrics = {
                    k: v for k, v in loss_dict.items() if k.startswith("test_")
                }
                if test_metrics:
                    logger.info(f"Test Control Metrics: {test_metrics}")

            return loss_dict

        if split in ["train"]:
            loss_dict = self.train_diffusion_forward(batch)
            return loss_dict

    def _get_cond_loss_weight(self):
        """Get current cond loss weight with optional warmup."""
        if self.cond_loss_warmup_steps <= 0:
            return self.cond_loss_weight

        if self.training_step < self.cond_loss_warmup_steps:
            # Linear warmup from 0 to target weight
            warmup_ratio = self.training_step / self.cond_loss_warmup_steps
            return self.cond_loss_weight * warmup_ratio
        else:
            return self.cond_loss_weight

    def set_training_step(self, step):
        """Set current training step for warmup/curriculum logic."""
        self.training_step = step

    def _get_timestep_weight(self, timesteps):
        """Get time-step dependent weight for cond loss."""
        if self.cond_loss_t_weight == "none":
            return torch.ones_like(timesteps, dtype=torch.float32)

        # Get alpha_bar values for the timesteps (ensure device compatibility)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(timesteps.device)
        alpha_bar = alphas_cumprod[timesteps]

        if self.cond_loss_t_weight == "alpha_bar":
            return alpha_bar
        elif self.cond_loss_t_weight == "sqrt_alpha_bar":
            return torch.sqrt(alpha_bar)
        elif self.cond_loss_t_weight == "snr":
            # Signal-to-noise ratio: alpha_bar / (1 - alpha_bar)
            snr = alpha_bar / (1 - alpha_bar + 1e-8)
            return snr / (snr + 1)  # Normalize to [0, 1]
        else:
            return torch.ones_like(timesteps, dtype=torch.float32)

    def compute_cond(self, cond_params):
        mean = torch.tensor(self.raw_mean, dtype=torch.float32)
        std = torch.tensor(self.raw_std, dtype=torch.float32)
        cond = torch.zeros([9, 3], dtype=torch.float32)
        cond_mask = torch.zeros([9], dtype=torch.bool)

        if "traj" in cond_params:
            cond[0] = torch.tensor(cond_params["traj"], dtype=torch.float32).reshape(3)
            cond[0, 1] = 0
            cond[0] = cond[0] * min(1.2, cond[0].norm()) / (cond[0].norm() + 1e-6)
            cond[0] = (cond[0] - mean[0]) / std[0]
            cond[0, 1] = 0
            cond_mask[0] = True

        if "heading" in cond_params:
            cond[1] = torch.tensor(cond_params["heading"], dtype=torch.float32).reshape(
                3
            )
            cond[1, 1] = 0
            cond[1] = cond[1] / cond[1].norm()
            cond_mask[1] = True

        if "vel" in cond_params:
            cond[2] = torch.tensor(cond_params["vel"], dtype=torch.float32).reshape(3)
            cond[2] = (cond[2] - mean[0]) / std[0]
            cond[2, 1] = 0
            cond_mask[2] = True

        if "pelvis" in cond_params:
            cond[3] = torch.tensor(cond_params["pelvis"], dtype=torch.float32).reshape(
                3
            )
            cond[3] = (cond[3] - mean[0]) / std[0]
            cond_mask[3] = True

        if "left_foot" in cond_params:
            cond[4] = torch.tensor(
                cond_params["left_foot"], dtype=torch.float32
            ).reshape(3)
            cond[4] = (cond[4] - mean[10]) / std[10]
            cond_mask[4] = True

        if "right_foot" in cond_params:
            cond[5] = torch.tensor(
                cond_params["right_foot"], dtype=torch.float32
            ).reshape(3)
            cond[5] = (cond[5] - mean[11]) / std[11]
            cond_mask[5] = True

        if "head" in cond_params:
            cond[6] = torch.tensor(cond_params["head"], dtype=torch.float32).reshape(3)
            cond[6] = (cond[6] - mean[15]) / std[15]
            cond_mask[6] = True

        if "left_wrist" in cond_params:
            cond[7] = torch.tensor(
                cond_params["left_wrist"], dtype=torch.float32
            ).reshape(3)
            cond[7] = (cond[7] - mean[20]) / std[20]
            cond_mask[7] = True

        if "right_wrist" in cond_params:
            cond[8] = torch.tensor(
                cond_params["right_wrist"], dtype=torch.float32
            ).reshape(3)
            cond[8] = (cond[8] - mean[21]) / std[21]
            cond_mask[8] = True

        return cond, cond_mask

    def t2m_generation(self, batch):
        texts = batch["y"]["text"]

        fixed_len = self.fixed_len
        latent_unit = self.latent_unit
        prefix_len = self.prefix_len
        complete_len = fixed_len - prefix_len
        complete_latent_len = complete_len // latent_unit
        batch_size = len(texts)

        sim_prefix = batch["y"]["sim_prefix"]
        gen_prefix = batch["y"]["gen_prefix"]

        conds = []
        cond_masks = []
        for cond_params in batch["cond_params"]:
            cond, cond_mask = self.compute_cond(cond_params)
            conds.append(cond)
            cond_masks.append(cond_mask)
        cond = torch.stack(conds).to(sim_prefix)
        cond_mask = torch.stack(cond_masks).to(sim_prefix)
        lengths = torch.tensor([60] * batch_size, dtype=torch.long).to(sim_prefix)

        sim_prefix = sim_prefix.squeeze(2).permute(0, 2, 1)
        gen_prefix = gen_prefix.squeeze(2).permute(0, 2, 1)

        if self.do_classifier_free_guidance:
            texts = texts + [""] * len(texts)
        text_emb = self.text_encoder(texts)

        prefix_mask = text_emb.new_ones([batch_size, prefix_len], dtype=torch.bool)
        mask = text_emb.new_ones([batch_size, fixed_len], dtype=torch.bool)

        sim_prefix_emb, _ = self.vae.encode_prefix(sim_prefix, prefix_mask)
        sim_prefix_emb = sim_prefix_emb * self.vae_scale_factor

        gen_prefix_emb, _ = self.vae.encode_prefix(gen_prefix, prefix_mask)
        gen_prefix_emb = gen_prefix_emb * self.vae_scale_factor

        latents = torch.randn(
            (batch_size, complete_latent_len, self.latent_dim[1]),
            device=text_emb.device,
        )

        # For diffusion, we use sim context (sim_prefix_emb)
        latents = self._diffusion_reverse(
            latents, sim_prefix_emb, text_emb, cond, cond_mask, lengths
        )
        # For vae, we use gen context (gen_prefix_emb)
        latents = torch.cat([gen_prefix_emb, latents], dim=1)
        latents = latents / self.vae_scale_factor

        feats_rst = self.vae.decode(latents, mask)
        return feats_rst
