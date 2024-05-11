# kaggle-hms

My solution for Kaggle competition: [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview)


## Solution
See `maxc_solution.md` for the solution overview.

## Work flow
Most of the components are in the `notebooks` directory.

Place the competition dataset under this directory.


## Training

note: You need to modify some paths in these notebook on your own environment.
See the instructions in each notebook.

1. `notebooks/gen_k_folds.ipynb`: Generate folds files.
2. `notebooks/preprocesss_dataset.ipynb`: Process datasets for model training.
3. `notebooks/train_spec_model.ipynb`: Train a model in the notebook. Change the model configs in the config block.

- (Optional): `notebooks/test_spec_dataset.ipynb`: Visualize the output of training dataloader.

- `scripts/train_model.py`: Run the script to train multiple models given a directory of configuration files. You can generate the configuration files using the first few blocks of `notebooks/train_spec_model.ipynb`. 

## Inference on Kaggle
- zip the pipeline and upload as a dataset: `zip -r hms_pipeline hms_pipline/`
- upload the checkpoints and config from the training output

see https://www.kaggle.com/code/asaliquid1011/hms-team-inference-ktmud#Max-Chen-Part for more details.
