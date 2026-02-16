import os
import numpy as np
from tqdm.auto import tqdm

import torch

from mld.config import parse_args, instantiate_from_config
from mld.models.modeltype.caldm import CALDM
from mld.data.humanml.scripts.motion_process import recover_from_ric
from mld.utils.utils import set_seed
from mld.data.humanml.utils.plot_script import plot_3d_motion
import types
from tqdm import tqdm


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


f2j_mean = np.load("data/mean.npy")
f2j_std = np.load("data/std.npy")


def feats2joints(features):
    mean = torch.tensor(f2j_mean).to(features)
    std = torch.tensor(f2j_std).to(features)
    features = features * std + mean
    return recover_from_ric(features, 22)


r_mean = np.load("data/mean.npy")
r_std = np.load("data/std.npy")
r_mean_eval = np.load("data/mean_eval.npy")
r_std_eval = np.load("data/std_eval.npy")


def renorm4t2m(features):
    # renorm to t2m norms for using t2m evaluators
    ori_mean = torch.tensor(r_mean).to(features)
    ori_std = torch.tensor(r_std).to(features)
    eval_mean = torch.tensor(r_mean_eval).to(features)
    eval_std = torch.tensor(r_std_eval).to(features)
    features = features * ori_std + ori_mean
    features = (features - eval_mean) / eval_std
    return features


def main():
    cfg = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(cfg.SEED_VALUE)

    cfg.output_dir = "tmp"
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/checkpoints", exist_ok=True)

    state_dict = torch.load(cfg.TRAIN.PRETRAINED, map_location="cpu")["state_dict"]

    pseudo_dataset = types.SimpleNamespace(
        feats2joints=feats2joints,
        renorm4t2m=renorm4t2m,
        is_mm=False,
    )
    cfg.DATASET.NFEATS = 263
    cfg.DATASET.NJOINTS = 22

    base_model = CALDM(cfg, pseudo_dataset)
    base_model.load_state_dict(state_dict)
    base_model.to(device)
    base_model.eval()

    texts = ["A person is jogging."]
    gen_len = 5

    prefix = torch.load("data/stance.pt")[0]
    prefix = prefix[:1].repeat(20, 0)[None].to(device)

    batch_size = len(texts)
    prefix = prefix.repeat(batch_size, 1, 1)
    texts = texts + [""] * len(texts)
    text_emb = base_model.text_encoder(texts)
    prefix_mask = torch.ones([batch_size, 20], dtype=torch.bool, device="cuda")
    mask = torch.ones([batch_size, 60], dtype=torch.bool, device="cuda")

    full_feats_rst = []

    for i in range(gen_len):
        prefix_emb, _ = base_model.vae.encode_prefix(prefix, prefix_mask)
        prefix_emb = prefix_emb * base_model.vae_scale_factor

        latents = torch.randn(
            (mask.shape[0], 10, base_model.latent_dim[1]),
            device=text_emb.device,
        )
        latents = base_model._diffusion_reverse(latents, prefix_emb, text_emb)

        latents = torch.cat([prefix_emb, latents], dim=1)
        latents = latents / base_model.vae_scale_factor
        feats_rst = base_model.vae.decode(latents, mask)
        full_feats_rst.append(feats_rst[:, 20:])
        prefix = feats_rst[:, -20:]

    feats_rst = torch.cat(full_feats_rst, dim=1)
    joints_rst = base_model.feats2joints(feats_rst)

    for i in tqdm(range(feats_rst.shape[0])):
        plot_3d_motion(
            f"video/{i}.mp4",
            joints_rst[i].detach().cpu().numpy(),
            texts[i],
            fps=20,
            hint=None,
        )


if __name__ == "__main__":
    main()
