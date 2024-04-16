# main python
from pytorch_lightning.core.optimizer import LightningOptimizer
import torch
import numpy as np
import os
import timm
from torch.optim.optimizer import Optimizer
import yaml
import gc
import pandas as pd
import pytorch_lightning as pl

from .spec_dataset import SpecDataset
from .eeg_model import SpecModel, SpecVitModel
from .kaggle_kl_div import eval_subm, eval_subm_g10


from typing import Any
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
import torch.nn as nn
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TQDMProgressBar,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb
import cv2



def log_images(loggers, key, images, log_size = 4):
    '''
        Utility function to log images to wandb
    '''
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            b = images.shape[0]
            log_size = min(b, log_size)
            log_imgs = []
            for i in range (log_size):
                log_imgs.append(images[i].detach().cpu().numpy())
            logger.log_image(key=key, images=log_imgs)


class SpecModule(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        # self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.val_step_pred = []
        self.val_step_eeg_id = []
        self.val_step_eeg_sub_id= []
        self.logged_train_img = False
        self.logged_val_img = False


        if config["use_ema"]:
            ema_decay = config["ema_decay"]
            self.ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
            print('Using EMA model with decay', ema_decay)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.parameters(), **self.config["optimizer_params"])

        if self.config["scheduler"]["name"] == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, **self.config["scheduler"]["params"]["CosineAnnealingLR"])
        elif self.config["scheduler"]["name"] == "CosineAnnealingWarmRestarts":
            scheduler = CosineAnnealingWarmRestarts(optimizer, **self.config["scheduler"]["params"]["CosineAnnealingWarmRestarts"])

        lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
    
    def get_kl_loss(self, output, target):
        kl_loss = nn.KLDivLoss(reduction="none")
        log_softmax_output = torch.log_softmax(output, dim=1)
        loss = kl_loss(log_softmax_output, target)
        loss = loss.sum(dim=1)
        return loss
    
    def training_step(self, batch, batch_idx):
        # specs, targets, metadata = batch["spec_img"], batch["label"], batch["metadata"]
        specs,  targets, metadata, soft_label, kl_weight = batch["spec_img"],batch["label"], batch["metadata"], batch["soft_label"], batch["kl_weight"]

        output = self.model(specs)
        log_softmax_output = torch.log_softmax(output, dim=1)


        loss = self.get_kl_loss(log_softmax_output, targets)
        loss = (loss * kl_weight).mean()
        self.log(f"train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # soft_loss = self.kl_loss(log_softmax_output, soft_label)
        # self.log(f"train_soft_loss", soft_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if not self.logged_train_img:
            log_images(self.trainer.loggers, "train_spec", specs)
            self.logged_train_img = True
        
        return loss
       
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.config["use_ema"]:
            self.ema_model.update_parameters(self.model)

    def validation_step(self, batch, batch_idx):
        specs,  targets, metadata, soft_label, kl_weight = batch["spec_img"],batch["label"], batch["metadata"], batch["soft_label"], batch["kl_weight"]

        if self.config["use_ema"]:
            output = self.ema_model(specs)
        else:
            output = self.model(specs)

        if torch.isnan(output).any():
            print(output)
            print(metadata)

        log_softmax_output = torch.log_softmax(output, dim=1)


        loss = self.get_kl_loss(log_softmax_output, targets)
        loss = (loss * kl_weight).mean()
        self.log(f"val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        output = torch.softmax(output, dim=1)
        self.val_step_pred.append(output.detach().cpu().numpy())
        self.val_step_eeg_id.append(metadata["eeg_id"].detach().cpu().numpy())
        self.val_step_eeg_sub_id.append(metadata["eeg_sub_id"].detach().cpu().numpy())


        if not self.logged_val_img:
            log_images(self.trainer.loggers, "val_spec", specs)
            self.logged_val_img = True

        return loss
    
    def on_validation_epoch_end(self) -> None:

        eeg_sub_ids = np.concatenate(self.val_step_eeg_sub_id)
        eeg_ids = np.concatenate(self.val_step_eeg_id)

        val_step_pred = np.concatenate(self.val_step_pred)

        subm_df = pd.DataFrame()
        subm_df["eeg_id"] = eeg_ids.astype(int)
        subm_df["eeg_sub_id"] = eeg_sub_ids.astype(int)
        TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
        subm_df[TARGETS] = val_step_pred

        output_dir = self.config["output_dir"]
        os.makedirs(f"{output_dir}/subm", exist_ok=True)
        subm_df.to_csv(f"{output_dir}/subm/val_subm.csv", index=False)
        
        gt_path = os.path.join(self.config["fold_dir"], "all_targets.csv")
        gt_df = pd.read_csv(gt_path)

        score = eval_subm(subm_df, gt_df)
        g10_score = eval_subm_g10(subm_df, gt_df)
        print("Score: ", score, "G10 Score: ", g10_score)
        epoch_num = self.current_epoch
        #  save 4 decimals score string
        current_fold = self.config["current_fold"]


        output_path = os.path.join(output_dir, f"subm/f{current_fold}_{epoch_num}_{score:.4f}_{g10_score:.4f}.csv")
        subm_df.to_csv(output_path, index=False)
        self.log(f"val_score", score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"val_g10_score", g10_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.val_step_meta = []
        self.val_step_pred = []
        self.val_step_eeg_id = []
        self.val_step_eeg_sub_id= []

def train_model(config_path, pretrained = True, ckpt_path = None, seed = 42):
    torch.set_float32_matmul_precision("medium")

    pl.seed_everything(seed)

    with open(config_path, "r") as file_obj:
        loaded_config = yaml.safe_load(file_obj)
    # Save config to new file
    # get base name of config file
    # create directory if it doesn't exist
    if not os.path.exists(loaded_config["output_dir"]):
        os.makedirs(loaded_config["output_dir"])
    config_name = os.path.basename(config_path)
    with open(os.path.join(loaded_config["output_dir"], config_name), "w") as file_obj:
        yaml.dump(loaded_config, file_obj, sort_keys=False)

    print(loaded_config)


    fold_dir = loaded_config["fold_dir"]
    folds = loaded_config["train_folds"]


    for fold in folds:
        print(f"Training fold {fold} ======================")
        run_name = (
            os.path.basename(os.path.normpath(loaded_config["output_dir"]))
            + f"_fold_{fold}"
        )

        fold_log_dir = os.path.join(loaded_config["output_dir"], f"fold_{fold}")
        wandb_logger = WandbLogger(
            save_dir=fold_log_dir,
            project=loaded_config["wandb_project"],
            name=run_name,
        )

        train_df_path = os.path.join(fold_dir, f"train_fold_{fold}.csv")
        val_df_path = os.path.join(fold_dir, f"val_fold_{fold}.csv")
        train_df = pd.read_csv(train_df_path)
        val_df = pd.read_csv(val_df_path)

        train_dataset = SpecDataset(
            train_df,
            configs=loaded_config,
            train=True,
        )
        val_dataset = SpecDataset(
            val_df,
            configs=loaded_config,
            train=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=loaded_config["train_bs"],
            num_workers=loaded_config["workers"],
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=loaded_config["valid_bs"],
            num_workers=loaded_config["workers"],
            shuffle=False,
        )
        print(len(train_loader), len(val_loader))

        val_loss_name = "val_score"
        g10_score_name = "val_g10_score"
        checkpoint_callback = ModelCheckpoint(
            save_weights_only=True,
            monitor=g10_score_name,
            dirpath=loaded_config["output_dir"],
            mode="min",
            filename=f"model-f{fold}-{{{g10_score_name}:.4f}}-{{{val_loss_name}:.4f}}",
            save_top_k=5,
            verbose=1,
        )

        progress_bar_callback = TQDMProgressBar()
        early_stopping_callback = EarlyStopping(
            monitor=g10_score_name, patience=loaded_config["patience"], verbose=1, mode="min"
        )

        wandb_logger.experiment.config.update(loaded_config)
        loggers = [CSVLogger(save_dir=fold_log_dir), wandb_logger]

        lr_callback = LearningRateMonitor(logging_interval="step")
        trainer = pl.Trainer(
            callbacks=[
                checkpoint_callback,
                early_stopping_callback,
                progress_bar_callback,
                lr_callback,
            ],
            logger=loggers,
            **loaded_config["trainer"],
        )
        loaded_config["current_fold"] = fold

        if loaded_config["scheduler"]["name"] == "CosineAnnealingLR":
            loaded_config["scheduler"]["params"]["CosineAnnealingLR"]["T_max"] = (
                len(train_loader) * loaded_config["trainer"]["max_epochs"]
            )

        if loaded_config["scheduler"]["name"] == "CosineAnnealingWarmRestarts":
            loaded_config["scheduler"]["params"]["CosineAnnealingWarmRestarts"]["T_0"] = (
                len(train_loader) * loaded_config["scheduler"]["params"]["CosineAnnealingWarmRestarts"]["T_0"]
            )

        if loaded_config["model_type"] == "SpecVitModel":
            print("Using SpecVitModel")
            model = SpecVitModel(loaded_config["spec_backbone"], vit_model = loaded_config["vit_model"], img_size = loaded_config["img_size"], pretrained=pretrained, feature_layer=loaded_config["feature_layer"],dropout=loaded_config["dropout"])
        elif loaded_config["model_type"] == "SpecEegModel":
            print("Using SpecEegModel")
            model = SpecEegModel(loaded_config["spec_backbone"], pretrained=pretrained, img_size = loaded_config["img_size"], dropout=loaded_config["dropout"])
        else:
            print("Using SpecModel")
            model = SpecModel(loaded_config["spec_backbone"], global_pool=loaded_config['global_pool'], pretrained=pretrained, img_size = loaded_config["img_size"], hidden_size=loaded_config["hidden_size"])

        module = SpecModule(model, loaded_config)
        img_size = loaded_config["img_size"]
        test_spec = torch.randn(1, 1, img_size[0], img_size[1])
        output = model(test_spec)
        print(output.shape)

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            module.load_state_dict(ckpt['state_dict'], strict=False)
        trainer.fit(module, train_loader, val_loader)
        # trainer.validate(module, val_loader)
        torch.cuda.empty_cache()
        gc.collect()

        wandb.finish()