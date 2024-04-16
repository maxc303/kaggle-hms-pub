import pandas as pd
import os
import torch
import sys
import glob

import yaml
import matplotlib.pyplot as plt

from .preproc import *
from .spec_dataset import SpecDataset, infer_spec_dataset
from .eeg_model import SpecModel, EegInferModule
from .kaggle_kl_div import eval_subm


def load_model_dir(model_dir):
    """
    Given a kaggle model directory, return the config path and the checkpoint paths
    """
    config_paths = glob.glob(f"{model_dir}/*.yaml")
    ckpt_paths = glob.glob(f"{model_dir}/*.ckpt")
    ckpt_paths.sort()

    assert len(config_paths) == 1
    return {"config_path": config_paths[0], "ckpt_paths": ckpt_paths}



def hms_inference(
    infer_df,
    model_dir_list,
    test_mode = True,
    tmp_dir="/kaggle/working/tmp",
    input_data_dir = "/kaggle/input/hms-harmful-brain-activity-classification/",
    verbose=False,
):
    
    # if infer df does not have eeg_sub_id, create it
    if "eeg_sub_id" not in infer_df.columns:
        infer_df["eeg_sub_id"] = 0
    if "eeg_label_offset_seconds" not in infer_df.columns:
        infer_df["eeg_label_offset_seconds"] = 0
    if "spectrogram_label_offset_seconds" not in infer_df.columns:
        infer_df["spectrogram_label_offset_seconds"] = 0

    if test_mode:
        EEG_DIR = os.path.join(input_data_dir, "test_eegs")
        LONG_SPEC_DIR = os.path.join(input_data_dir, "test_spectrograms")
    else:
        EEG_DIR = os.path.join(input_data_dir, "train_eegs")
        LONG_SPEC_DIR = os.path.join(input_data_dir, "train_spectrograms")

    if verbose:
        print("EEG_DIR", EEG_DIR)
        print("LONG_SPEC_DIR", LONG_SPEC_DIR)

    EEG_SPEC_NPY_DIR = os.path.join(tmp_dir, "eeg_specs")
    LONG_SPEC_NPY_DIR = os.path.join(tmp_dir, "long_specs")
    RAW_EEG_NPY_DIR = os.path.join(tmp_dir, "raw_eegs")

    os.makedirs(EEG_SPEC_NPY_DIR, exist_ok=True)
    os.makedirs(LONG_SPEC_NPY_DIR, exist_ok=True)
    os.makedirs(RAW_EEG_NPY_DIR, exist_ok=True)

    eeg_ids = infer_df.eeg_id.unique().tolist()
    spec_ids = infer_df.spectrogram_id.unique().tolist()

    ### Preprocess dataset
    preprocess_raw_eeg_list(
        EEG_DIR,
        eeg_ids,
        RAW_EEG_NPY_DIR,
        filter_type="sosbandclip",
        sub_width=2500,
        num_workers=-1,
        norm_type="constant",
    )
    preprocess_long_spec_list(
        LONG_SPEC_DIR, spec_ids, LONG_SPEC_NPY_DIR, 100, num_workers=-1
    )
    preprocess_eeg_spec_list(
        EEG_DIR,
        eeg_ids,
        EEG_SPEC_NPY_DIR,
        signal_height=100,
        sub_width=250,
        gen_freq=5,
        prefilter=False,
        num_workers=-1,
        n_fft=1024,
        win_length=200,
        lowcut=0.5,
    )

    ### Read model config and checkpoint
    MODEL_META_LIST = []

    for model_dir in model_dir_list:
        MODEL_META_LIST.append(load_model_dir(model_dir))

    if verbose:
        for model_meta in MODEL_META_LIST:
            print("-----", model_meta["config_path"])
            for ckpt_path in model_meta["ckpt_paths"]:
                print(ckpt_path)

    ### Augmentation list for TTAs
    infer_ds_aug_list = [
        {
            "lrflip_prob": 0.0,
            "fbflip_prob": 0.0,
            "mask_prob": 0.0,
            "keep_center_ratio": 0.0,
            "hflip_prob": 0.0,
            "blur_prob": 0.0,
            "roll_prob": 0.0,
            "neg_eeg_prob": 0.0,
            "contrast_prob": 0.0,
            "fuse_prob": 0.0,
            "block_prob": 0.0,
            "noise_prob": 0.0,
            "mask_iter": 5,
            "mask_size_ratio": 0.15,
            "num_block_ch": 4,
            "dummy_votes_prob": 0.0,
            "num_dummy_votes": 1,
        },
        {
            "lrflip_prob": 1.0,
            "fbflip_prob": 0.0,
            "mask_prob": 0.0,
            "keep_center_ratio": 0.0,
            "hflip_prob": 0.0,
            "blur_prob": 0.0,
            "roll_prob": 0.0,
            "neg_eeg_prob": 0.0,
            "contrast_prob": 0.0,
            "fuse_prob": 0.0,
            "block_prob": 0.0,
            "noise_prob": 0.0,
            "mask_iter": 5,
            "mask_size_ratio": 0.15,
            "num_block_ch": 4,
            "dummy_votes_prob": 0.0,
            "num_dummy_votes": 1,
        },
    ]

    ### Main inference loop
    all_result_dfs = []
    all_result_logits_dfs = []
    for model_meta in MODEL_META_LIST:
        config_path = model_meta["config_path"]
        with open(config_path, "r") as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)

        model_config["raw_eeg_dir"] = RAW_EEG_NPY_DIR
        model_config["eeg_spec_dir"] = EEG_SPEC_NPY_DIR
        model_config["long_spec_dir"] = LONG_SPEC_NPY_DIR

        model_config["val_dataset_group"] = "eeg_id"
        model_config["filter_type"] = ""

        if verbose:
            print(model_config)

        for aug_probs in infer_ds_aug_list:
            model_config["val_aug_probs"] = aug_probs

            infer_ds = SpecDataset(
                infer_df,
                model_config,
                train=False,
            )
            if verbose:
                print(len(infer_ds))
                sample = infer_ds[0]["spec_img"]
                plt.figure(figsize=(5, 5))
                plt.imshow(sample[0])
                plt.show()

            for ckpt_path in model_meta["ckpt_paths"]:
                model = SpecModel(
                    model_config["spec_backbone"],
                    global_pool=model_config["global_pool"],
                    pretrained=False,
                    img_size=model_config["img_size"],
                )
                module = EegInferModule(model, model_config)
                ckpt = torch.load(ckpt_path)
                module.load_state_dict(ckpt["state_dict"], strict=False)

                result_df, result_logits_df = infer_spec_dataset(eeg_dataset=infer_ds, model=module)
                all_result_dfs.append(result_df)
                all_result_logits_dfs.append(result_logits_df)

    ### Ensemble results
    ensemble_result_df = pd.DataFrame()
    ensemble_result_logits_df = pd.DataFrame()
    target_columns = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    subm_eeg_id = None
    subm_eeg_sub_id = None
    subm_pred = None
    subm_pred_logits = None
    for i in range(len(all_result_dfs)):
        result_df = all_result_dfs[i]
        result_logits_df = all_result_logits_dfs[i]
        if subm_eeg_id is None:
            subm_eeg_id = result_df["eeg_id"]
        if subm_eeg_sub_id is None:
            subm_eeg_sub_id = result_df["eeg_sub_id"]
        if subm_pred is None:
            subm_pred = result_df[target_columns] / len(all_result_dfs)
            subm_pred_logits = result_logits_df[target_columns] / len(all_result_dfs)
        else:
            subm_pred[target_columns] += result_df[target_columns] / len(all_result_dfs)
            subm_pred_logits[target_columns] += result_logits_df[target_columns] / len(all_result_dfs)

    ensemble_result_df["eeg_id"] = subm_eeg_id
    ensemble_result_df["eeg_sub_id"] = subm_eeg_sub_id
    ensemble_result_df[target_columns] = subm_pred

    ensemble_result_logits_df["eeg_id"] = subm_eeg_id
    ensemble_result_logits_df["eeg_sub_id"] = subm_eeg_sub_id
    ensemble_result_logits_df[target_columns] = subm_pred_logits

    subm_columns = [
        "eeg_id",
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    subm_df = ensemble_result_df[subm_columns]

    subm_logits_df = ensemble_result_logits_df[subm_columns]    
    if test_mode:
        return subm_df, subm_logits_df
    
    return ensemble_result_df, ensemble_result_logits_df

