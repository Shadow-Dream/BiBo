import os
import numpy as np
from tqdm.auto import tqdm

import torch
import time
from mld.config import parse_args, instantiate_from_config
from mld.models.modeltype.cacldm import CACLDM
from mld.data.humanml.scripts.motion_process import recover_from_ric
from mld.utils.utils import set_seed
from mld.data.humanml.utils.plot_script import plot_3d_motion
import types
from tqdm import tqdm

f2j_mean = np.load("data/mean.npy")
f2j_std = np.load("data/std.npy")
r_mean = np.load("data/mean.npy")
r_std = np.load("data/std.npy")
r_mean_eval = np.load("data/mean_eval.npy")
r_std_eval = np.load("data/std_eval.npy")
raw_mean = np.load("data/Mean_raw.npy")
raw_std = np.load("data/Std_raw.npy")
def compute_cond(cond_params):
    mean = torch.tensor(raw_mean,dtype=torch.float32)
    std = torch.tensor(raw_std,dtype=torch.float32)
    cond = torch.zeros([9,3],dtype=torch.float32)
    cond_mask = torch.zeros([9],dtype=torch.bool)

    if "traj" in cond_params:
        cond[0] = torch.tensor(cond_params["traj"],dtype=torch.float32).reshape(3)
        cond[0,1] = 0
        cond[0] = cond[0] * min(1.2, cond[0].norm()) / (cond[0].norm() + 1e-6)
        cond[0] = (cond[0] - mean[0]) / std[0]
        cond[0,1] = 0
        cond_mask[0] = True
    
    if "heading" in cond_params:
        cond[1] = torch.tensor(cond_params["heading"],dtype=torch.float32).reshape(3)
        cond[1,1] = 0
        cond[1] = cond[1] / cond[1].norm()
        cond_mask[1] = True

    if "vel" in cond_params:
        cond[2] = torch.tensor(cond_params["vel"],dtype=torch.float32).reshape(3)
        cond[2] = (cond[2] - mean[0]) / std[0]
        cond[2,1] = 0
        cond_mask[2] = True

    if "pelvis" in cond_params:
        cond[3] = torch.tensor(cond_params["pelvis"],dtype=torch.float32).reshape(3)
        cond[3] = (cond[3] - mean[0]) / std[0]
        cond_mask[3] = True

    if "left_foot" in cond_params:
        cond[4] = torch.tensor(cond_params["left_foot"],dtype=torch.float32).reshape(3)
        cond[4] = (cond[4] - mean[10]) / std[10]
        cond_mask[4] = True

    if "right_foot" in cond_params:
        cond[5] = torch.tensor(cond_params["right_foot"],dtype=torch.float32).reshape(3)
        cond[5] = (cond[5] - mean[11]) / std[11]
        cond_mask[5] = True

    if "head" in cond_params:
        cond[6] = torch.tensor(cond_params["head"],dtype=torch.float32).reshape(3)
        cond[6] = (cond[6] - mean[15]) / std[15]
        cond_mask[6] = True

    if "left_wrist" in cond_params:
        cond[7] = torch.tensor(cond_params["left_wrist"],dtype=torch.float32).reshape(3)
        cond[7] = (cond[7] - mean[20]) / std[20]
        cond_mask[7] = True

    if "right_wrist" in cond_params:
        cond[8] = torch.tensor(cond_params["right_wrist"],dtype=torch.float32).reshape(3)
        cond[8] = (cond[8] - mean[21]) / std[21]
        cond_mask[8] = True

    return cond, cond_mask

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def feats2joints(features):
    mean = torch.tensor(f2j_mean).to(features)
    std = torch.tensor(f2j_std).to(features)
    features = features * std + mean
    return recover_from_ric(features, 22)

def renorm4t2m(features):
    # renorm to t2m norms for using t2m evaluators
    ori_mean = torch.tensor(r_mean).to(features)
    ori_std = torch.tensor(r_std).to(features)
    eval_mean = torch.tensor(r_mean_eval).to(features)
    eval_std = torch.tensor(r_std_eval).to(features)
    features = features * ori_std + ori_mean
    features = (features - eval_mean) / eval_std
    return features

@torch.no_grad()
def main():
    cfg = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(cfg.SEED_VALUE)

    cfg.output_dir = "tmp"
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/checkpoints", exist_ok=True)

    state_dict = torch.load(cfg.TRAIN.PRETRAINED, map_location="cpu")["state_dict"]

    pseudo_dataset = types.SimpleNamespace(
        feats2joints = feats2joints,
        renorm4t2m = renorm4t2m,
        is_mm = False,
    )
    cfg.DATASET.NFEATS = 263
    cfg.DATASET.NJOINTS = 22

    base_model = CACLDM(cfg, pseudo_dataset).to(torch.float32)
    base_model.load_state_dict(state_dict)
    base_model.to(device)

    base_model.eval()

    prefix = torch.load("data/stance.pt")[0]
    prefix = prefix[:1].repeat(20, 1)[None].to(device)
    
    lengths = torch.tensor([60],dtype=torch.long).to(device)
    batch_size = 1

    prefix_mask = torch.ones([batch_size, 20], dtype=torch.bool, device="cuda")
    mask = torch.ones([batch_size, 60], dtype=torch.bool, device="cuda")

    full_feat_rst = []

    texts = ["a person is dancing.", ""]
    cond, cond_mask = compute_cond({})
    text_emb = base_model.text_encoder(texts)
    cond = cond.to(device).unsqueeze(0)
    cond_mask = cond_mask.to(device).unsqueeze(0)

    times = []

    for i in range(10000):
        start_time = time.time()
        prefix_emb, _ = base_model.vae.encode_prefix(prefix, prefix_mask)
        prefix_emb = prefix_emb * base_model.vae_scale_factor

        latents = torch.randn(
            (mask.shape[0], 10, base_model.latent_dim[1]),
            device=text_emb.device,
        )
        latents = base_model._diffusion_reverse(latents, prefix_emb, text_emb, cond, cond_mask, lengths)

        latents = torch.cat([prefix_emb, latents], dim=1)
        latents = latents / base_model.vae_scale_factor
        feats_rst = base_model.vae.decode(latents, mask)
        end_time = time.time()
        times.append(end_time - start_time)
        feats_rst = feats_rst[:,20:]
        full_feat_rst.append(feats_rst)
        prefix = feats_rst[:,-20:]
        print("Time: ", sum(times) / len(times), sum(times), 40*len(times))

    feats_rst = torch.cat(full_feat_rst, dim=1)
    joints_rst = base_model.feats2joints(feats_rst)
    plot_3d_motion(f"video/0.mp4", joints_rst[0].detach().cpu().numpy(), texts[0], fps=20, hint=None)

if __name__ == "__main__":
    main()
