
# main python
import torch
import numpy as np
import os
import timm
import yaml
import gc
import pandas as pd
import pytorch_lightning as pl

import argparse

import sys
sys.path.append('../')
sys.path.append('../hms_pipeline')

from hms_pipeline.trainer import train_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config_dir', type=str, help='Training configuration directory', required=True)
    parser.add_argument('--dry_run', action='store_true', help='Dry run (just print the config files to be used for training)')
    args = parser.parse_args()


    # list all files in the directory
    files = os.listdir(args.config_dir)

    config_paths = [os.path.join(args.config_dir, file) for file in files]
    # sort the files by name
    config_paths.sort()
    print(config_paths)

    for config_path in config_paths:
        if args.dry_run:
            print(f'Dry run: {config_path}')
        else:
            train_model(config_path)
            # try:
            #     train_model(config_path)
            # except Exception as e:
            #     print(f'Error in training model {config_path}: {e}')
            #     continue