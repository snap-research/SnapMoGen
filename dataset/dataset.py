import collections

import torch
import numpy as np
from torch.utils import data
from os.path import join as pjoin
import random
from tqdm import tqdm
import json

# from utils.paramUtil import style_enumerator, style_inv_enumerator


class CommonMotionDataset(data.Dataset):
    def __init__(self, cfg, mean, std, mid_list_path, cid_list_path):
        self.cfg = cfg
        mid_list = []
        cid_list = []
        total_frames = 0

        data_dict = {}

        with open(mid_list_path, "r") as f:
            for line in f.readlines():
                mid_list.append(line.strip())

        with open(cid_list_path, "r") as f:
            for line in f.readlines():
                cid = line.strip()
                _, start, end = cid.split("#")

                if int(end) - int(start) >= cfg.data.min_motion_length:
                    cid_list.append(cid)
                    total_frames += int(end) - int(start)

        # for fid in fids_list:

        total_count = len(cid_list)

        for i, mid in tqdm(enumerate(mid_list)):
            data_path = pjoin(cfg.data.feat_dir, "%s.npy" % mid)
            data = np.load(data_path)
            data_dict[mid] = data

        # if cfg.is_train and (not fix_bias):
        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.cfg = cfg
        self.mid_list = mid_list
        self.cid_list = cid_list

        print(
            "Loading %d motions, %d frames, %03f hours"
            % (total_count, total_frames, total_frames / 30.0 / 60.0 / 60.0)
        )
        # print("Loading %d style motions, %d style frames, %03f hours"%(num_style_motions, total_style_frames, total_style_frames/30./60./60.))

    def inv_transform(self, data):
        if isinstance(data, np.ndarray):
            return data * self.std[:data.shape[-1]] + self.mean[:data.shape[-1]]
        elif isinstance(data, torch.Tensor):
            return data * torch.from_numpy(self.std[:data.shape[-1]]).float().to(
                data.device
            ) + torch.from_numpy(self.mean[:data.shape[-1]]).float().to(data.device)
        else:
            raise TypeError("Expected data to be either np.ndarray or torch.Tensor")

    def __len__(self):
        return len(self.cid_list)

    def __getitem__(self, item):
        cid = self.cid_list[item]
        mid, start, end = cid.split("#")
        motion = self.data_dict[mid][int(start) : int(end)]

        # Z Normalization
        motion_data = (motion - self.mean) / self.std

        # print(self.std)

        return motion_data, cid


class TextMotionDataset(CommonMotionDataset):
    def __init__(self, cfg, mean, std, mid_list_path, cid_list_path, all_caption_path):
        super().__init__(cfg, mean, std, mid_list_path, cid_list_path)

        with open(all_caption_path, "r") as f:
            self.all_captions = json.load(f)

    def __getitem__(self, item):
        motion, cid = super().__getitem__(item)
        captions = self.all_captions[cid]["manual"] + self.all_captions[cid]["gpt"]
        caption = random.choice(captions)
        m_length = (
            len(motion)
            if len(motion) < self.cfg.data.max_motion_length
            else self.cfg.data.max_motion_length
        )

        # coin2 = np.random.choice(["single", "single", "double"])
        # if coin2 == "double":
        #     m_length = (
        #         m_length // self.cfg.data.unit_length - 1
        #     ) * self.cfg.data.unit_length
        # else:
        m_length = (
                m_length // self.cfg.data.unit_length
            ) * self.cfg.data.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx : idx + m_length]
        if m_length < self.cfg.data.max_motion_length:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros(
                        (self.cfg.data.max_motion_length - m_length, motion.shape[1])
                    ),
                ],
                axis=0,
            )

        return caption, motion, m_length


class MotionDataset(CommonMotionDataset):
    def __init__(self, cfg, mean, std, mid_list_path, cid_list_path):
        super().__init__(cfg, mean, std, mid_list_path, cid_list_path)
        lengths = [0]
        n_cid_list = []
        for cid in self.cid_list:
            _, start, end = cid.split("#")
            length = int(end) - int(start) - self.cfg.data.motion_length
            if length >= 0:
                lengths.append(length)
                n_cid_list.append(cid)

        self.cid_list = n_cid_list
        self.cumsum = np.cumsum(lengths)

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        cid_idx = np.searchsorted(self.cumsum, item + 1) - 1
        # cid =
        idx = item - self.cumsum[cid_idx]
        motion, _ = super().__getitem__(cid_idx)
        motion_clip = motion[idx : idx + self.cfg.data.motion_length]

        return motion_clip
