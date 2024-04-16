import torch
import numpy as np
import torchvision.transforms.v2 as T
import albumentations as A
import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
import cv2

from .project_configs import TARGETS, SIGNAL_NAME, EEG_FEATS

from scipy.signal import butter, lfilter


def load_data_meta(data_path):
    metadata_path = os.path.join(data_path, "metadata.yaml")
    metadata = yaml.load(open(metadata_path, "r"), Loader=yaml.FullLoader)
    return metadata


def load_raw_eeg_npy(raw_eeg_dir, eeg_id, offset_sec, channel_width=4, timespan = 10):
    metadata = load_data_meta(raw_eeg_dir)
    raw_eeg_path = os.path.join(raw_eeg_dir, f"{eeg_id}.npy")
    raw_eeg_dict = np.load(raw_eeg_path, allow_pickle=True).item()
    freq = metadata["freq"]
    len = metadata["sub_width"]
    assert int(freq * 50) == len

    offset_sec_unit = int(offset_sec * freq)

    raw_full_eeg_img_dict = dict()
    raw_eeg_img_dict = dict()
    # for k, v in raw_eeg_dict.items():
    #     print(k, v.shape)

    eeg_start = offset_sec_unit
    eeg_len = 50 * freq

    assert timespan%2 ==0, "Timespan should be even, got {}".format(timespan)
    assert timespan <= 50

    center_start = offset_sec_unit + (25 - timespan//2) * freq
    center_len = timespan * freq

    for key, value in raw_eeg_dict.items():
        signal_data = value.values
        center_data = signal_data[center_start : center_start + center_len].T
        full_data = signal_data[eeg_start : eeg_start + eeg_len].T

        # full_data = np.repeat(full_data, channel_width, axis=0)
        # center_data = np.repeat(center_data, channel_width, axis=0)
        raw_eeg_img_dict[key] = center_data.copy()
        raw_full_eeg_img_dict[key] = full_data.copy()

    return raw_eeg_img_dict, raw_full_eeg_img_dict


def repeat_raw_eeg_dict(raw_eeg_dict, repeat=4):
    for k, v in raw_eeg_dict.items():
        raw_eeg_dict[k] = np.repeat(v, repeat, axis=0)
    return raw_eeg_dict


def load_eeg_spec_npy(eeg_spec_dir, eeg_id, offset_sec):
    metadata = load_data_meta(eeg_spec_dir)
    freq = metadata["freq"]
    len = metadata["sub_width"]
    assert int(freq * 50) == len

    offset_sec_unit = int(offset_sec * freq)
    eeg_spec_path = os.path.join(eeg_spec_dir, f"{eeg_id}.npy")
    eeg_spec_dict = np.load(eeg_spec_path, allow_pickle=True).item()

    center_start = offset_sec_unit + 20 * freq
    center_len = 10 * freq

    center_eeg_spec_dict = dict()

    for k, v in eeg_spec_dict.items():
        eeg_spec_dict[k] = v[:, offset_sec_unit : offset_sec_unit + len].copy()
        center_eeg_spec_dict[k] = v[:, center_start : center_start + center_len].copy()
    return eeg_spec_dict, center_eeg_spec_dict


def load_long_spec_npy(long_spec_dir, long_spec_id, offset_sec):
    metadata = load_data_meta(long_spec_dir)
    freq = metadata["freq"]
    len = metadata["sub_width"]
    assert int(freq * 600) == len

    offset_sec_unit = int(offset_sec * freq)
    long_spec_path = os.path.join(long_spec_dir, f"{long_spec_id}.npy")
    long_spec_dict = np.load(long_spec_path, allow_pickle=True).item()

    for k, v in long_spec_dict.items():
        long_spec_dict[k] = v[:, offset_sec_unit : offset_sec_unit + len]
    return long_spec_dict

def get_partial_long_spec(long_spec_dict, ratio = 0.5):
    # get the center ratio of the long spec

    for k, v in long_spec_dict.items():
        start = int(v.shape[1] * (1 - ratio) / 2)
        end = int(v.shape[1] * (1 + ratio) / 2)
        long_spec_dict[k] = v[:, start:end]
    return long_spec_dict

def get_sub_eeg_spec(eeg_spec_dict, offset_sec=0, hop_length=20):
    # get the 50 sec eeg spec for the sub id
    num_row_per_sec = 200 // hop_length
    col_start = int(offset_sec * num_row_per_sec)

    for k, v in eeg_spec_dict.items():
        # crop 50 sec
        eeg_spec_dict[k] = v[:, col_start : col_start + num_row_per_sec * 50]
    return eeg_spec_dict


def random_mask_channel(raw_eeg_dict, p=0.5, block_prob=0.5, num_block=4):
    if p is None or np.random.rand() >= p:
        return raw_eeg_dict

    # randomly select 4 index from 0 to 15
    block_idx = np.random.choice(16, num_block, replace=False)

    idx = 0
    for k, v in raw_eeg_dict.items():
        for i in range(v.shape[0]):
            if idx in block_idx:
                v[i, :] = 0
            idx += 1
    return raw_eeg_dict

def flip_fb(spec_dict):
    '''
    Randomly flip the eeg from front to back by reordering channels
    '''
    for k, v in spec_dict.items():
            spec_dict[k] = cv2.flip(spec_dict[k], 0)
    return spec_dict

def random_mask_dict(spec_dict, p=0.5, max_size_ratio=0.2, iter=3, time_only=False, keep_center_ratio = 0):
    if p is None or np.random.rand() >= p:
        return spec_dict
    for i in range(iter):
        start_frac = np.random.rand()
        size_frac = np.random.rand() * max_size_ratio
        
        
        for k, v in spec_dict.items():
            start_idx = int(start_frac * v.shape[1])
            end_idx = int(start_idx + size_frac * v.shape[1])
            end_idx = min(end_idx, v.shape[1])
            before_mask = v.copy()
            spec_dict[k][:, start_idx:end_idx] = 0

            if keep_center_ratio > 0:
                start_idx = int((1-keep_center_ratio) * v.shape[1]/2)
                end_idx = int(start_idx + keep_center_ratio * v.shape[1])
                # spec_dict[k][:, start_idx:end_idx] = 1
                spec_dict[k][:, start_idx:end_idx] = before_mask[:, start_idx:end_idx]


        if not time_only:
            start_frac = np.random.rand()
            size_frac = np.random.rand() * max_size_ratio
            for k, v in spec_dict.items():
                start_idx = int(start_frac * v.shape[0])
                end_idx = int(start_idx + size_frac * v.shape[0])
                end_idx = min(end_idx, v.shape[0])
                spec_dict[k][start_idx:end_idx, :] = 0
    
    return spec_dict

def random_blur(img, p = 0.5):
    kernel_sizes = [3, 5, 7, 9, 11]
    if p is None or np.random.rand() >= p:
        return img

    kernel_size = np.random.choice(kernel_sizes)
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img

############################################
# Image Generation
############################################


def dict_to_img_vertical(
    img_dict,
    width=128,
    height=512,
    signals=["LL", "LP", "RL", "RP"],
    vflip=[False, False, False, False],
):
    num_signals = len(signals)
    signal_height = height / num_signals
    assert height % num_signals == 0
    img = np.zeros((height, width))
    for idx, signal in enumerate(signals):
        eeg_signal_img = img_dict[signal]
        if vflip[idx]:
            eeg_signal_img = cv2.flip(eeg_signal_img, 0)

        eeg_signal_img = cv2.resize(eeg_signal_img, (width, int(signal_height)))

        img[idx * int(signal_height) : (idx + 1) * int(signal_height), :] = (
            eeg_signal_img
        )

    return img


def raw_eeg_dict_to_224_img(
    raw_eeg_dict,
    signals=["LL", "LP", "RL", "RP"],
    vflip=[False, False, False, False],
):
    imgs = []

    for idx, signal in enumerate(signals):
        eeg_signal_img = raw_eeg_dict[signal]
        if vflip[idx]:
            eeg_signal_img = cv2.flip(eeg_signal_img, 0)
        imgs.append(eeg_signal_img)
    img = np.hstack(imgs)
    # resize img to 16 * 3584
    img = cv2.resize(img, (3584, 16))
    # reorder the img to 224 * 224 row by row

    square_img = np.zeros((224, 224))
    for i in range(224 // 16):
        square_img[i * 16 : (i + 1) * 16, :] = img[:, i * 224 : (i + 1) * 224]

    return square_img


default_aug_probs = {
    "lrflip_prob": 0.0,
    "mask_prob": 0.0,
    "hflip_prob": 0.0,
    "roll_prob": 0.0,
    "neg_eeg_prob": 0.0,
    "fuse_prob": 0.0,
    "block_prob": 0.0,
    "noise_prob": 0.0,
}


def load_imgs_from_meta(
    sample_metadata,
    configs,
    aug_probs=default_aug_probs,
):
    eeg_sec_offset = sample_metadata["eeg_label_offset_seconds"]
    long_spec_offset = sample_metadata["spectrogram_label_offset_seconds"]

    eeg_spec_dir = configs["eeg_spec_dir"]
    long_spec_dir = configs["long_spec_dir"]
    raw_eeg_dir = configs["raw_eeg_dir"]
    img_types = configs["img_types"]
    img_size = configs["img_size"]
    sub_image_size = configs["sub_img_size"]
    sub_img_vflips = configs["sub_img_vflips"]

    center_timespan = 10
    if "center_timespan" in configs:
        center_timespan = configs["center_timespan"]
    raw_full_gap = 32
    if "raw_full_gap" in configs:
        raw_full_gap = configs["raw_full_gap"]

    signals = configs["signals"]
    if np.random.rand() < aug_probs["lrflip_prob"]:
        signals = configs["lrflip_signals"]

    final_img = np.zeros(img_size)
    eeg_spec_img = np.zeros(sub_image_size["eeg_spec"])
    long_spec_img = np.zeros(sub_image_size["long_spec"])
    raw_eeg_img = np.zeros(sub_image_size["raw_eeg"])
    full_raw_eeg_img = np.zeros(sub_image_size["full_raw_eeg"])

    if "eeg_spec" in img_types or "ceeg_spec" in img_types:
        vflips = sub_img_vflips["eeg_spec"]
        eeg_spec_dict, ceeg_spec_dict = load_eeg_spec_npy(
            eeg_spec_dir, sample_metadata["eeg_id"], eeg_sec_offset
        )

        if "ceeg_spec" in img_types:
            eeg_spec_dict = random_mask_dict(ceeg_spec_dict, aug_probs["mask_prob"], iter = aug_probs["mask_iter"], max_size_ratio= aug_probs["mask_size_ratio"])
            eeg_spec_img = dict_to_img_vertical(
                ceeg_spec_dict,
                width=sub_image_size["eeg_spec"][1],
                height=sub_image_size["eeg_spec"][0],
                signals=signals,
                vflip=vflips,
            )
        else:
            eeg_spec_dict = random_mask_dict(eeg_spec_dict, aug_probs["mask_prob"], iter = aug_probs["mask_iter"], max_size_ratio= aug_probs["mask_size_ratio"], keep_center_ratio=aug_probs["keep_center_ratio"])
            eeg_spec_img = dict_to_img_vertical(
                eeg_spec_dict,
                width=sub_image_size["eeg_spec"][1],
                height=sub_image_size["eeg_spec"][0],
                signals=signals,
                vflip=vflips,
            )

    if "long_spec" in img_types:
        vflips = sub_img_vflips["long_spec"]

        long_spec_dict = load_long_spec_npy(
            long_spec_dir, sample_metadata["spectrogram_id"], long_spec_offset
        )

        long_spec_dict = get_partial_long_spec(long_spec_dict, ratio = configs["long_spec_ratio"])
        long_spec_dict = random_mask_dict(long_spec_dict, aug_probs["mask_prob"], iter = aug_probs["mask_iter"], max_size_ratio= aug_probs["mask_size_ratio"])
        long_spec_img = dict_to_img_vertical(
            long_spec_dict,
            width=sub_image_size["long_spec"][1],
            height=sub_image_size["long_spec"][0],
            signals=signals,
            vflip=vflips,
        )

    if "raw_eeg" in img_types or "full_raw_eeg" in img_types:
        vflips = sub_img_vflips["raw_eeg"]
        center_raw_eeg_dict, full_raw_eeg_dict = load_raw_eeg_npy(
            raw_eeg_dir, sample_metadata["eeg_id"], eeg_sec_offset, timespan=center_timespan
        )

        if np.random.rand() < aug_probs["fbflip_prob"]:
            center_raw_eeg_dict = flip_fb(center_raw_eeg_dict)
            full_raw_eeg_dict = flip_fb(full_raw_eeg_dict)

        center_raw_eeg_dict = random_mask_dict(
            center_raw_eeg_dict, aug_probs["mask_prob"], time_only=True, iter = aug_probs["mask_iter"], max_size_ratio= aug_probs["mask_size_ratio"]
        )
        full_raw_eeg_dict = random_mask_dict(
            full_raw_eeg_dict, aug_probs["mask_prob"], time_only=True, iter= aug_probs["mask_iter"], max_size_ratio= aug_probs["mask_size_ratio"]
        )

        center_raw_eeg_dict = random_mask_channel(
            center_raw_eeg_dict, aug_probs["mask_prob"], block_prob=aug_probs["block_prob"], num_block=aug_probs["num_block_ch"]
        )

        signal_pixel = (sub_image_size["raw_eeg"][0] + 15) // 16
        center_raw_eeg_dict = repeat_raw_eeg_dict(
            center_raw_eeg_dict, repeat=signal_pixel
        )

        raw_eeg_img = dict_to_img_vertical(
            center_raw_eeg_dict,
            width=sub_image_size["raw_eeg"][1],
            height=sub_image_size["raw_eeg"][0],
            signals=signals,
            vflip=vflips,
        )
        full_raw_eeg_dict = repeat_raw_eeg_dict(full_raw_eeg_dict, repeat=signal_pixel)
        full_raw_eeg_img = dict_to_img_vertical(
            full_raw_eeg_dict,
            width=sub_image_size["full_raw_eeg"][1],
            height=sub_image_size["full_raw_eeg"][0],
            signals=signals,
            vflip=vflips,
        )
       
        if np.random.rand() < aug_probs["neg_eeg_prob"]:
            raw_eeg_img = -raw_eeg_img
            full_raw_eeg_img = -full_raw_eeg_img
            # generate a random number from 0.5 to 1.5

        if np.random.rand() < aug_probs["contrast_prob"]:
            contrast = np.random.rand()  + 0.5
            raw_eeg_img = raw_eeg_img * contrast
            full_raw_eeg_img = full_raw_eeg_img * contrast
            
        if np.random.rand() < aug_probs["roll_prob"]:
            roll_dist = np.random.randint(0, raw_eeg_img.shape[1])
            raw_eeg_img = np.roll(raw_eeg_img, roll_dist, axis=1)

    # long_spec_img = random_blur(long_spec_img, p = aug_probs["blur_prob"])
    eeg_spec_img = random_blur(eeg_spec_img, p = aug_probs["blur_prob"])

    if np.random.rand() < aug_probs["hflip_prob"]:
        eeg_spec_img = cv2.flip(eeg_spec_img, 1)
        long_spec_img = cv2.flip(long_spec_img, 1)
        raw_eeg_img = cv2.flip(raw_eeg_img, 1)
        full_raw_eeg_img = cv2.flip(full_raw_eeg_img, 1)

    # plate the long spec on the top left
    if "long_spec" in img_types:
        if np.random.rand() < aug_probs["contrast_prob"]:
            long_contrast = np.random.rand()  + 0.5
            long_spec_img = long_spec_img * long_contrast
        final_img[: long_spec_img.shape[0], : long_spec_img.shape[1]] = long_spec_img
    # place the image on the top right
    if "eeg_spec" in img_types or "ceeg_spec" in img_types:
        if np.random.rand() < aug_probs["contrast_prob"]:
            eeg_contrast = np.random.rand()  + 0.5
            eeg_spec_img = eeg_spec_img * eeg_contrast

        final_img[: eeg_spec_img.shape[0], -eeg_spec_img.shape[1] :] = eeg_spec_img
    # place the raw eeg on the bottom
    if "raw_eeg" in img_types:
        final_img[-raw_eeg_img.shape[0] :, : raw_eeg_img.shape[1]] = raw_eeg_img
    # place the full raw eeg on top of the raw eeg
        


    if "full_raw_eeg" in img_types:
        if "raw_eeg" in img_types:
            gap = raw_full_gap
            final_img[
                -raw_eeg_img.shape[0] - full_raw_eeg_img.shape[0] -gap : -raw_eeg_img.shape[0]-gap,
                : full_raw_eeg_img.shape[1],
            ] = full_raw_eeg_img
        else:
            final_img[-full_raw_eeg_img.shape[0] :, : full_raw_eeg_img.shape[1]] = full_raw_eeg_img

    # final_img = raw_eeg_img
    if np.random.rand() < aug_probs["noise_prob"]:
        # generate a random gaussian noise
        noise = np.random.normal(0, 1, final_img.shape)
        final_img = final_img + noise

    return final_img


def soften_label(label, soft_value=0.1):
    # label_sum = np.sum(label)
    label = label.astype(np.float32)
    label += soft_value
    label = label / np.sum(label)
    return label


def target_to_kl_weight(target_tensor, mean_pred = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]):
    assert len(mean_pred) == 6
    assert len(mean_pred) == len(target_tensor)
    mean_pred = torch.tensor(mean_pred)
    kl_loss = torch.nn.functional.kl_div(
        mean_pred.log(), target_tensor, reduction="none"
    )
    kl_loss = kl_loss.sum(dim=0)
    return kl_loss

def dummy_votes(targets, num_votes = 1):
    # the input is a 1d array of targets with size 6
    # randomly add votes to the targets where the value is already greater than 1
    # find the indices where the targets are greater than 1
    indices = np.where(targets >= 1)[0]

    # randomly select the indices using the num_votes
    selected_indices = np.random.choice(indices, num_votes, replace=False)
    for idx in selected_indices:
        targets[idx] += 1
    return targets

def remove_votes(targets, num_votes = 1):
    # the input is a 1d array of targets with size 6
    # randomly add votes to the targets where the value is already greater than 1
    # find the indices where the targets are greater than 1
    indices = np.where(targets > 1)[0]

    # randomly select the indices using the num_votes
    selected_indices = np.random.choice(indices, num_votes, replace=False)
    for idx in selected_indices:
        if targets[idx] > 1:
            targets[idx] -= 1
    return targets


class SpecDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        eeg_df,
        configs,
        train=False,
    ):

        self.configs = configs
        if train:
            self.sub_sample_group = self.configs["train_dataset_group"]
            self.aug_probs = self.configs["train_aug_probs"]
        else:
            self.sub_sample_group = self.configs["val_dataset_group"]
            self.aug_probs = self.configs["val_aug_probs"]

        self.eeg_df = eeg_df

        self.eeg_df_list = []

        # if not train, all the aug probs should be either 0 or 1
        if not train:
            for k, v in self.aug_probs.items():
                # assert and print the key
                if "prob" in k:
                    assert v == 0 or v >= 1.0, f"Aug prob {k} should be 0 or 1.0"
        print(self.aug_probs)

        # sub sample for only getting one sample per eeg in each epoch
        if self.sub_sample_group is not None:
            for eeg_id, eeg_meta_df in eeg_df.groupby(self.sub_sample_group):
                self.eeg_df_list.append(eeg_meta_df)

        # check if TARGETS are not in the columns, set inference mode
        self.inference = False
        if not set(TARGETS).issubset(eeg_df.columns):
            print(
                "Provided dataframe does not contain targets, setting inference mode to True."
            )
            self.inference = True
        self.train = train

    def __len__(self):
        if self.sub_sample_group is not None:
            return len(self.eeg_df_list)

        return len(self.eeg_df)

    def get_single_item(self, idx):
        if self.sub_sample_group is not None:
            eeg_meta_df = self.eeg_df_list[idx]
            sample_idx = 0
            if self.train:
                # sample a sub eeg from the group
                sample_idx = np.random.choice(len(eeg_meta_df), 1)[0]
            row = eeg_meta_df.iloc[sample_idx]
        else:
            row = self.eeg_df.iloc[idx]

        metadata = row.to_dict()

        spec_img = load_imgs_from_meta(
            metadata,
            configs=self.configs,
            aug_probs=self.aug_probs,
        )

        if self.inference:
            label = np.zeros((6,))
        else:
            targets = row[TARGETS].values
            # print(targets)
            if np.random.rand() < self.aug_probs["dummy_votes_prob"]:
                targets = dummy_votes(targets, self.aug_probs["num_dummy_votes"])
            # print("Targets after dummy votes", targets)
            targets = targets.astype(np.float32)

            target_sum = targets.sum()
            label = targets / target_sum

        if self.configs["inverse_kl_weight"]:
            kl_multiplier = self.configs["kl_multiplier"]
            kl_weight = 1/ (target_to_kl_weight(torch.tensor(label), mean_pred=self.configs["inverse_kl_mean"]) * kl_multiplier)
        else:
            # torch scalar 1
            kl_weight = torch.tensor(1.0)

        if (not self.inference) and self.configs["vote_weight"]:
            total_votes = metadata["total_votes"]
            weight = (total_votes/(total_votes+1))
            kl_weight *= (weight * weight)

        if (not self.inference) and metadata["total_votes"] <=7:
            kl_weight*=self.configs["l7_weight"]


        spec_img = torch.tensor(spec_img).float()
        spec_img = spec_img.unsqueeze(0)
        soft_label = soften_label(label, 0.1 / 6.0)
        return {
            "spec_img": spec_img,
            "label": label,
            "metadata": metadata,
            "soft_label": soft_label,
            "kl_weight": kl_weight,
        }

    def __getitem__(self, idx):
        first_item = self.get_single_item(idx)
        if np.random.rand() < self.aug_probs["fuse_prob"]:
            # generate a random negative eeg
            second_item = self.get_single_item(np.random.choice(len(self)))
            first_item_weight = 0.75
            second_item_weight = 0.25
            fused_item = first_item
            # long spec size
            long_spec_size = self.configs["sub_img_size"]["long_spec"]

            fused_item["spec_img"] = first_item["spec_img"]
            # replace the long spec with the second item long spec
            fused_item["spec_img"][:, :long_spec_size[0], :long_spec_size[1]] = second_item["spec_img"][:, : long_spec_size[0], :long_spec_size[1]]

            fused_item["label"] = (
                first_item["label"] * first_item_weight
                + second_item["label"] * second_item_weight
            )
            fused_item["soft_label"] = (
                first_item["soft_label"] * first_item_weight
                + second_item["soft_label"] * second_item_weight
            )
            fused_item["metadata"]["fused_id"] = second_item["metadata"]["eeg_id"]

            # recalculate kl weight
            if self.configs["inverse_kl_weight"]:
                kl_multiplier = self.configs["kl_multiplier"]
                kl_weight = 1/ (target_to_kl_weight(torch.tensor(fused_item["label"])) * kl_multiplier)
            else:
                # torch scalar 1
                kl_weight = torch.tensor(1.0)

            if (not self.inference) and first_item["metadata"]["total_votes"] <=7:
                kl_weight*=self.configs["l7_weight"]

            fused_item["kl_weight"] = kl_weight

        else:
            fused_item = first_item
            fused_item["metadata"]["fused_id"] = 0
        return fused_item


def infer_spec_dataset(eeg_dataset, model, batch_size=32, device="cuda"):
    dataloader = torch.utils.data.DataLoader(
        eeg_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    model = model.eval()
    model = model.to(device)

    all_preds = []
    all_logits = []
    all_eeg_ids = []
    all_eeg_sub_ids = []
    for batch in tqdm(dataloader):
        spec = batch["spec_img"].to(device)
        eeg_id = batch["metadata"]["eeg_id"]
        eeg_sub_id = batch["metadata"]["eeg_sub_id"]
        # raw_eeg_img = batch["raw_eeg_img"].to(device)
        with torch.no_grad():
            preds = model(spec)

        all_logits.append(preds.cpu().numpy())
        preds = torch.softmax(preds, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_eeg_ids.append(eeg_id.cpu().numpy())
        all_eeg_sub_ids.append(eeg_sub_id.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_eeg_ids = np.concatenate(all_eeg_ids, axis=0)
    all_eeg_sub_ids = np.concatenate(all_eeg_sub_ids, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    df = pd.DataFrame()
    df["eeg_id"] = all_eeg_ids
    df["eeg_sub_id"] = all_eeg_sub_ids
    df[TARGETS] = all_preds

    logits_df = pd.DataFrame()
    logits_df["eeg_id"] = all_eeg_ids
    logits_df["eeg_sub_id"] = all_eeg_sub_ids
    logits_df[TARGETS] = all_logits

    return df, logits_df
