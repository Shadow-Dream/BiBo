import os
import random
import logging
import codecs as cs
from os.path import join as pjoin

import numpy as np
from rich.progress import track

import torch
from torch.utils.data import Dataset

from .scripts.motion_process import recover_from_ric
from .utils.word_vectorizer import WordVectorizer

logger = logging.getLogger(__name__)


class MotionDataset(Dataset):
    def __init__(
        self,
        mean,
        std,
        split_file,
        motion_dir,
        window_size,
        tiny=False,
        progress_bar=True,
        **kwargs,
    ):
        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        maxdata = 10 if tiny else 1e10
        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split_file.split('/')[-1].split('.')[0]}",
                )
            )
        else:
            enumerator = enumerate(id_list)

        count = 0
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name + ".npy"))
                if motion.shape[0] < window_size:
                    continue
                self.lengths.append(motion.shape[0] - window_size)
                self.data.append(motion)
            except Exception as e:
                print(e)
                pass

        self.cumsum = np.cumsum([0] + self.lengths)
        if not tiny:
            logger.info(
                "Total number of motions {}, snippets {}".format(
                    len(self.data), self.cumsum[-1]
                )
            )

        self.mean = mean
        self.std = std
        self.window_size = window_size

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx : idx + self.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std
        return motion, self.window_size


class Text2MotionDataset(Dataset):
    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        fps,
        padding_to_max,
        njoints,
        tiny=False,
        progress_bar=True,
        fixed_len=0,
        has_postfix=False,
        postfix_len=4,
        postfix_type="stance",
        has_prefix=True,
        postfix_sampling="normal",
        **kwargs,
    ):
        self.mean = mean
        self.std = std
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.padding_to_max = padding_to_max
        self.njoints = njoints
        self.is_test = "test" in split_file
        self.has_postfix = has_postfix
        self.postfix_sampling = postfix_sampling
        self.postfix_len = postfix_len
        if self.is_test:
            self.has_postfix = False
            self.fixed_len = 0
            self.padding_to_max = True
            self.max_motion_length = 200
            is_stance = False
        else:
            self.fixed_len = fixed_len
            is_stance = has_prefix

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list

        maxdata = 100 if tiny else 1e10
        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split_file.split('/')[-1].split('.')[0]}",
                )
            )
        else:
            enumerator = enumerate(id_list)

        if is_stance:
            stance_motion = torch.load("stance.pt")[0]
            stance_motion = stance_motion[:1]
            stance_motion = stance_motion.cpu().numpy() * self.std + self.mean

            self.postfix_type = postfix_type
            self.prefix_motion = stance_motion.repeat(20, 0)
            self.postfix_motion = stance_motion.repeat(postfix_len, 0)
            if postfix_type == "zero":
                self.postfix_motion = (
                    np.zeros_like(self.postfix_motion) * self.std + self.mean
                )

        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name + ".npy"))
                if (
                    len(motion) < self.min_motion_length
                    or len(motion) >= self.max_motion_length
                ):
                    bad_count += 1
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * fps) : int(to_tag * fps)]
                                if (len(n_motion)) < self.min_motion_length or len(
                                    n_motion
                                ) >= self.max_motion_length:
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                    + "_"
                                    + name
                                )
                                while new_name in data_dict:
                                    new_name = (
                                        random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                        + "_"
                                        + name
                                    )
                                if is_stance:
                                    if has_postfix:
                                        n_motion = np.concatenate(
                                            [
                                                self.prefix_motion,
                                                n_motion,
                                                self.postfix_motion,
                                            ]
                                        )
                                    else:
                                        n_motion = np.concatenate(
                                            [self.prefix_motion, n_motion]
                                        )

                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except ValueError:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if flag:
                    if is_stance:
                        if has_postfix:
                            motion = np.concatenate(
                                [self.prefix_motion, motion, self.postfix_motion]
                            )
                        else:
                            motion = np.concatenate([self.prefix_motion, motion])

                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    count += 1
            except Exception as e:
                print(e)
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1])
        )

        if not tiny:
            logger.info(f"Reading {len(self.id_list)} motions from {split_file}.")
            logger.info(f"Total {len(name_list)} motions are used.")
            logger.info(
                f"{bad_count} motion sequences not within the length range of "
                f"[{self.min_motion_length}, {self.max_motion_length}) are filtered out."
            )

        control_args = kwargs["control_args"]
        self.control_mode = None
        if os.path.exists(control_args.MEAN_STD_PATH):
            self.raw_mean = np.load(pjoin(control_args.MEAN_STD_PATH, "Mean_raw.npy"))
            self.raw_std = np.load(pjoin(control_args.MEAN_STD_PATH, "Std_raw.npy"))
        else:
            self.raw_mean = self.raw_std = None
        if not tiny and control_args.CONTROL:
            self.t_ctrl = control_args.TEMPORAL
            self.training_control_joints = np.array(control_args.TRAIN_JOINTS)
            self.testing_control_joints = np.array(control_args.TEST_JOINTS)
            self.training_density = control_args.TRAIN_DENSITY
            self.testing_density = control_args.TEST_DENSITY

            self.control_mode = (
                "val" if ("test" in split_file or "val" in split_file) else "train"
            )
            if self.control_mode == "train":
                logger.info(f"Training Control Joints: {self.training_control_joints}")
                logger.info(f"Training Control Density: {self.training_density}")
            else:
                logger.info(f"Testing Control Joints: {self.testing_control_joints}")
                logger.info(f"Testing Control Density: {self.testing_density}")
            logger.info(f"Temporal Control: {self.t_ctrl}")

        self.data_dict = data_dict
        self.name_list = name_list

    def __len__(self):
        return len(self.name_list)

    def random_mask(self, joints):
        choose_joint = self.testing_control_joints

        length = joints.shape[0]
        density = self.testing_density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)

        if self.t_ctrl:
            choose_seq = np.arange(0, choose_seq_num)
        else:
            choose_seq = np.random.choice(length, choose_seq_num, replace=False)
            choose_seq.sort()

        mask_seq = np.zeros((length, self.njoints, 3))
        for cj in choose_joint:
            mask_seq[choose_seq, cj] = 1.0

        joints = (joints - self.raw_mean) / self.raw_std
        joints = joints * mask_seq
        return joints, mask_seq

    def random_mask_train(self, joints):
        if self.t_ctrl:
            choose_joint = self.training_control_joints
        else:
            num_joints = len(self.training_control_joints)
            num_joints_control = 1
            choose_joint = np.random.choice(
                num_joints, num_joints_control, replace=False
            )
            choose_joint = self.training_control_joints[choose_joint]

        length = joints.shape[0]

        if self.training_density == "random":
            choose_seq_num = np.random.choice(length - 1, 1) + 1
        else:
            choose_seq_num = int(
                length
                * random.uniform(self.training_density[0], self.training_density[1])
                / 100
            )

        if self.t_ctrl:
            choose_seq = np.arange(0, choose_seq_num)
        else:
            choose_seq = np.random.choice(length, choose_seq_num, replace=False)
            choose_seq.sort()

        mask_seq = np.zeros((length, self.njoints, 3))
        for cj in choose_joint:
            mask_seq[choose_seq, cj] = 1

        joints = (joints - self.raw_mean) / self.raw_std
        joints = joints * mask_seq
        return joints, mask_seq

    def __getitem__(self, idx):
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[: self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        m_length = (m_length // self.unit_length) * self.unit_length

        if self.fixed_len > 0:
            m_length = min(m_length, self.fixed_len)

        if self.has_postfix:
            if self.postfix_sampling == "normal":
                idx = random.randint(0, max(len(motion) - m_length, 0))
            elif self.postfix_sampling == "one":
                max_start = max(len(motion) - m_length - self.postfix_len + 1, 0)
                idx = random.randint(0, max_start)
                if idx == max_start:
                    idx += random.randint(0, self.postfix_len - 1)
        else:
            idx = random.randint(0, max(len(motion) - m_length, 0))
        motion = motion[idx : idx + m_length]
        m_length = motion.shape[0]

        hint, hint_mask = None, None
        if self.control_mode is not None:
            joints = recover_from_ric(torch.from_numpy(motion).float(), self.njoints)
            joints = joints.numpy()
            if self.control_mode == "train":
                hint, hint_mask = self.random_mask_train(joints)
            else:
                hint, hint_mask = self.random_mask(joints)

            if self.padding_to_max:
                padding = np.zeros((self.max_motion_length - m_length, *hint.shape[1:]))
                hint = np.concatenate([hint, padding], axis=0)
                hint_mask = np.concatenate([hint_mask, padding], axis=0)

        "Z Normalization"
        motion = (motion - self.mean) / self.std
        if self.fixed_len > 0:
            padding = np.zeros((self.fixed_len - m_length, motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)
        elif self.padding_to_max:
            padding = np.zeros((self.max_motion_length - m_length, motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            (hint, hint_mask),
        )
