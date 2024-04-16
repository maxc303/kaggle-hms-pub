# kaggle-hms

My solution for Kaggle competition: [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview)


## Solution
See `maxc_solution.md` for the solution overview.

## Work flow
Most of the components are in the `notebooks` directory.

Place the competition dataset under this directory.


## Training
1. `notebooks/gen_k_folds.ipynb`: Generate folds files.
2. `notebooks/preprocesss_dataset.ipynb`: Process datasets for model training.
3. `notebooks/train_spec_model.ipynb`: Train a model in the notebook. Change the model configs in the config block.


- `scripts/train_model.py`: Run the script to train multiple models given a directory of configuration files.

## Inference on Kaggle
- zip the pipeline and upload as a dataset: `zip -r hms_pipeline hms_pipline/`
- upload the checkpoints and config from the training output
