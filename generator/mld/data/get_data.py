from typing import Optional
from os.path import join as pjoin

import numpy as np

from omegaconf import DictConfig

from .data import DataModule
from .base import BaseDataModule
from .utils import mld_collate, mld_collate_motion_only
from .humanml.utils.word_vectorizer import WordVectorizer


def get_mean_std(phase, cfg, dataset_name):
    name = "t2m" if dataset_name == "humanml3d" else dataset_name
    assert name in ["t2m", "kit"]
    if phase in ["val"]:
        if name == "t2m":
            data_root = pjoin(cfg.model.t2m_path, name, "Comp_v6_KLD01", "meta")
        elif name == "kit":
            data_root = pjoin(cfg.model.t2m_path, name, "Comp_v6_KLD005", "meta")
        else:
            raise ValueError("Only support t2m and kit")
        mean = np.load(pjoin(data_root, "mean.npy"))
        std = np.load(pjoin(data_root, "std.npy"))
    else:
        data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
        mean = np.load(pjoin(data_root, "Mean.npy"))
        std = np.load(pjoin(data_root, "Std.npy"))

    return mean, std


def get_WordVectorizer(cfg, dataset_name):
    if dataset_name.lower() in ["humanml3d", "kit"]:
        return WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")
    else:
        raise ValueError("Only support WordVectorizer for HumanML3D and KIT")


dataset_module_map = {"humanml3d": DataModule, "kit": DataModule}
motion_subdir = {"humanml3d": "new_joint_vecs", "kit": "new_joint_vecs"}


def get_dataset(cfg, motion_only=False):
    dataset_name = cfg.DATASET.NAME
    if dataset_name.lower() in ["humanml3d", "kit"]:
        data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
        mean, std = get_mean_std("train", cfg, dataset_name)
        mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)
        wordVectorizer = None if motion_only else get_WordVectorizer(cfg, dataset_name)
        collate_fn = mld_collate_motion_only if motion_only else mld_collate
        dataset = dataset_module_map[dataset_name.lower()](
            name=dataset_name.lower(),
            cfg=cfg,
            motion_only=motion_only,
            collate_fn=collate_fn,
            mean=mean,
            std=std,
            mean_eval=mean_eval,
            std_eval=std_eval,
            w_vectorizer=wordVectorizer,
            text_dir=pjoin(data_root, "texts"),
            motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
            max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
            min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
            max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
            unit_length=eval(f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
            fps=eval(f"cfg.DATASET.{dataset_name.upper()}.FRAME_RATE"),
            padding_to_max=cfg.DATASET.PADDING_TO_MAX,
            window_size=cfg.DATASET.WINDOW_SIZE,
            fixed_len=cfg.DATASET.FIXED_LEN if hasattr(cfg.DATASET, "FIXED_LEN") else 0,
            control_args=eval(f"cfg.DATASET.{dataset_name.upper()}.CONTROL_ARGS"),
            has_postfix=(
                cfg.DATASET.HAS_POSTFIX
                if hasattr(cfg.DATASET, "HAS_POSTFIX")
                else False
            ),
            postfix_len=(
                cfg.DATASET.POSTFIX_LEN if hasattr(cfg.DATASET, "POSTFIX_LEN") else 4
            ),
            postfix_type=(
                cfg.DATASET.POSTFIX_TYPE
                if hasattr(cfg.DATASET, "POSTFIX_TYPE")
                else "stance"
            ),
            has_prefix=(
                cfg.DATASET.HAS_PREFIX if hasattr(cfg.DATASET, "HAS_PREFIX") else True
            ),
            postfix_sampling=(
                cfg.DATASET.POSTFIX_SAMPLING
                if hasattr(cfg.DATASET, "POSTFIX_SAMPLING")
                else "normal"
            ),
        )

        cfg.DATASET.NFEATS = dataset.nfeats
        cfg.DATASET.NJOINTS = dataset.njoints
        return dataset

    elif dataset_name.lower() in ["humanact12", "uestc", "amass"]:
        raise NotImplementedError
