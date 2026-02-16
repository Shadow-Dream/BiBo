import time
import inspect
import logging

import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler

from mld.config import instantiate_from_config
from mld.utils.temos_utils import lengths_to_mask, remove_padding
from mld.utils.utils import (
    count_parameters,
    get_guidance_scale_embedding,
    extract_into_tensor,
)
from mld.data.humanml.utils.plot_script import plot_3d_motion

from .base import BaseModel
import torch.nn as nn

logger = logging.getLogger(__name__)


class CALDM(BaseModel):
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

        if cfg.model.guidance_scale == "dynamic":
            s_cfg = cfg.model.scheduler
            self.guidance_scale = s_cfg.cfg_step_map[s_cfg.num_inference_steps]
            logger.info(f"Guidance Scale set as {self.guidance_scale}")

        self.text_encoder = instantiate_from_config(cfg.model.text_encoder)
        self.vae = instantiate_from_config(cfg.model.motion_vae)
        self.denoiser = instantiate_from_config(cfg.model.denoiser)

        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.alphas = torch.sqrt(self.scheduler.alphas_cumprod)
        self.sigmas = torch.sqrt(1 - self.scheduler.alphas_cumprod)

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

        self.dno = (
            instantiate_from_config(cfg.model["noise_optimizer"])
            if cfg.model.get("noise_optimizer")
            else None
        )

        self.summarize_parameters()

    @property
    def do_classifier_free_guidance(self):
        return self.guidance_scale > 1

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
        self,
        latents,
        prefix,
        encoder_hidden_states,
    ):
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

    def _diffusion_process(self, latents, encoder_hidden_states):
        prefix = latents[:, : self.prefix_latent_len, :]
        latents = latents[:, self.prefix_latent_len :, :]

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
            dtype=torch.long,
        )

        noisy_latents = self.scheduler.add_noise(latents.clone(), noise, timesteps)

        model_output = self.denoiser(
            sample=noisy_latents,
            prefix=prefix,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        )

        latents_pred, noise_pred = self.predicted_origin(
            model_output, timesteps, noisy_latents
        )

        n_set = {
            "noise": noise,
            "noise_pred": noise_pred,
            "sample_pred": latents_pred,
            "sample_gt": latents,
        }
        return n_set

    def train_diffusion_forward(self, batch):
        feats_ref = batch["motion"]
        mask = batch["mask"]
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, mask)
        z = z * self.vae_scale_factor

        text = batch["text"]
        text = ["" if np.random.rand(1) < self.guidance_uncondp else i for i in text]
        text_emb = self.text_encoder(text)
        n_set = self._diffusion_process(z, text_emb)

        loss_dict = dict()

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

        total_loss = sum(loss_dict.values())
        loss_dict["loss"] = total_loss
        return loss_dict

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

        sim_prefix = sim_prefix.squeeze(2).permute(0, 2, 1)
        gen_prefix = gen_prefix.squeeze(2).permute(0, 2, 1)

        if self.do_classifier_free_guidance:
            texts = texts + [""] * len(texts)
        text_emb = self.text_encoder(texts)
        if not hasattr(self, "save_index"):
            self.save_index = 0
        # torch.save(sim_prefix, f"tmp/sim_prefix{self.save_index}.pt")
        # torch.save(gen_prefix, f"tmp/gen_prefix{self.save_index}.pt")
        # torch.save(text_emb, f"tmp/text_emb{self.save_index}.pt")
        self.save_index += 1

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
        latents = self._diffusion_reverse(latents, sim_prefix_emb, text_emb)
        # For vae, we use gen context (gen_prefix_emb)
        latents = torch.cat([gen_prefix_emb, latents], dim=1)
        latents = latents / self.vae_scale_factor

        feats_rst = self.vae.decode(latents, mask)
        # mean = torch.tensor(self.datamodule.hparams['mean']).to(feats_rst)
        # std = torch.tensor(self.datamodule.hparams['std']).to(feats_rst)
        # feats_rst = feats_rst * std + mean
        # torch.save(feats_rst, f"tmp/feats_rst{self.save_index}.pt")
        return feats_rst

    def t2m_eval(self, batch):
        texts = batch["text"]
        full_feats_ref = batch["motion"]
        full_mask = batch["mask"]
        full_len = 196
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

        if self.do_classifier_free_guidance:
            texts = texts + [""] * len(texts)
        text_emb = self.text_encoder(texts)

        iterations = (full_len - 1) // complete_len + 1
        prefix = full_feats_ref[:, :prefix_len]

        if getattr(self.cfg.VAL, "FIXED_PREFIX", False):
            mean = torch.tensor(
                self.datamodule.train_dataset.mean, device=full_feats_ref.device
            )
            std = torch.tensor(
                self.datamodule.train_dataset.std, device=full_feats_ref.device
            )
            prefix = torch.tensor(
                self.datamodule.train_dataset.prefix_motion,
                device=full_feats_ref.device,
            )
            prefix = (prefix - mean) / std
            prefix = prefix[0]
            prefix = prefix[None, None].repeat(batch_size, prefix_len, 1)

        full_feats_rst = []
        for i in range(iterations):
            prefix_mask = torch.ones(
                [batch_size, prefix_len], dtype=torch.bool, device="cuda"
            )
            mask = torch.ones([batch_size, fixed_len], dtype=torch.bool, device="cuda")

            prefix_emb, _ = self.vae.encode_prefix(prefix, prefix_mask)
            prefix_emb = prefix_emb * self.vae_scale_factor

            latents = torch.randn(
                (mask.shape[0], complete_latent_len, self.latent_dim[1]),
                device=text_emb.device,
            )
            latents = self._diffusion_reverse(latents, prefix_emb, text_emb)

            latents = torch.cat([prefix_emb, latents], dim=1)
            latents = latents / self.vae_scale_factor
            feats_rst = self.vae.decode(latents, mask)
            full_feats_rst.append(feats_rst[:, prefix_len:])

            prefix = feats_rst[:, -prefix_len:]

        full_feats_rst = torch.cat(full_feats_rst, dim=1)
        feats_ref = full_feats_ref[:, :full_len]
        feats_rst = full_feats_rst[:, :full_len]

        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(feats_ref)

        # for i in range(batch_size):
        #     motion = joints_rst[i]
        #     motion_index = len(os.listdir("videos"))
        #     plot_3d_motion(f"videos/{motion_index}.mp4", motion.detach().cpu().numpy(), texts[i], fps=20)

        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        feats_ref = self.datamodule.renorm4t2m(feats_ref)

        # use original frame lengths and align everything with the same order
        lengths_tensor = torch.tensor(lengths, device=feats_ref.device)
        align_idx = np.argsort(lengths_tensor.data.tolist())[::-1].copy()

        feats_rst = feats_rst[align_idx]
        feats_ref = feats_ref[align_idx]
        joints_rst = joints_rst[align_idx]
        joints_ref = joints_ref[align_idx]
        orig_lengths = lengths_tensor[align_idx]

        # motion/text embeddings computed with length-based align_idx
        m_lens = torch.div(orig_lengths, self.latent_unit, rounding_mode="floor")
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
        return rs_set

    def allsplit_step(self, split, batch):
        if split in ["test", "val"]:
            rs_set = self.t2m_eval(batch)
            if rs_set is None:
                return None
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
                elif metric == "PhysicalMetrics":
                    getattr(self, metric).update(
                        rs_set["joints_rst"],
                        batch["length"],
                    )
                else:
                    raise TypeError(f"Not support this metric: {metric}.")
            return None

        if split in ["train"]:
            loss_dict = self.train_diffusion_forward(batch)
            return loss_dict
