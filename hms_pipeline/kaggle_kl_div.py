import numpy as np
import pandas as pd
import pandas.api.types

# import kaggle_metric_utilities

from . import kaggle_metric_utilities
from typing import Optional


class ParticipantVisibleError(Exception):
    pass

def team_eval(df_subm, df_labels_all_path = "/home/maxc/workspace/kaggle-hms/notebooks/train_fold_irr_mark.csv"):
    df_labels_all_path = "/home/maxc/workspace/kaggle-hms/notebooks/train_fold_irr_mark.csv"
    df_labels_all = pd.read_csv(df_labels_all_path)
    df_subm = df_subm.rename({'seizure_vote':'seizure_pred','lpd_vote':'lpd_pred','gpd_vote':'gpd_pred','lrda_vote':'lrda_pred',
                                    'grda_vote':'grda_pred','other_vote':'other_pred',
                                    }, axis=1)
    df_subm = df_subm.merge(df_labels_all[['eeg_id','eeg_sub_id','label_id','patient_id',
                                                'seizure_vote','lpd_vote','gpd_vote','lrda_vote','grda_vote','other_vote',
                                                'fold','n_votes']], on=['eeg_id','eeg_sub_id'], how='left')
    
    def get_team_score(df_subm, calc_mode = 'high_vote'):
        if calc_mode  == 'high_vote':
            df_subm   = df_subm[df_subm['n_votes']>9].reset_index(drop=True)
        # df_oof_chen   = df_subm.groupby('eeg_id').first().reset_index(drop=True)
        df_oof =  df_subm.groupby('eeg_id').apply(lambda x: x.iloc[len(x) // 2]).reset_index(drop=True) 
        col_labels = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
        df_labels           = df_oof.copy()
        df_labels           = df_labels[col_labels]
        df_labels['id']     = np.arange(len(df_labels))


        df_oof         = df_oof[['seizure_pred','lpd_pred','gpd_pred','lrda_pred','grda_pred','other_pred',]]
        df_oof         = df_oof.rename({'seizure_pred':'seizure_vote','lpd_pred':'lpd_vote','gpd_pred':'gpd_vote','lrda_pred':'lrda_vote',
                                                'grda_pred':'grda_vote','other_pred':'other_vote',
                                                    }, axis=1)
        df_oof['id']   = np.arange(len(df_oof))


        print(df_labels.shape, df_oof.shape)
        metrics_flg1_vote   = score(solution=df_labels.copy(), submission=df_oof.copy(), 
                                                                        row_id_column_name='id')
        return metrics_flg1_vote
    
    all_score = get_team_score(df_subm, calc_mode = 'all')
    high_score = get_team_score(df_subm, calc_mode = 'high_vote')
    print(all_score, high_score)

def eval_subm(subm_df, gt_df):
    subm_df = subm_df.copy()
    gt_df = gt_df.copy()
    
    gt_df['eeg_id'] = gt_df['eeg_id'].astype(int)
    gt_df['eeg_sub_id'] = gt_df['eeg_sub_id'].astype(int)
    subm_df['eeg_id'] = subm_df['eeg_id'].astype(int)
    subm_df['eeg_sub_id'] = subm_df['eeg_sub_id'].astype(int)

    # create unique id for each row from eeg_id and eeg_sub_id
    gt_df['unique_id'] = gt_df['eeg_id'].astype(str) + '_' + gt_df['eeg_sub_id'].astype(str)
    subm_df['unique_id'] = subm_df['eeg_id'].astype(str) + '_' + subm_df['eeg_sub_id'].astype(str)


    TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    subm_df = subm_df[['unique_id'] + TARGETS]
    gt_df = gt_df[['unique_id'] + TARGETS]
    old_gt_df_shape = gt_df.shape
    subm_unique_ids = subm_df['unique_id']
    gt_df = gt_df[gt_df['unique_id'].isin(subm_unique_ids)]

    print(old_gt_df_shape, gt_df.shape, subm_df.shape)
    
    assert gt_df.shape[0] == subm_df.shape[0], f"Submission and ground truth have different number of unique ids: {gt_df.shape[0]} vs {subm_df.shape[0]}"
    # sort gt_df and subm_df by unique_id
    gt_df.sort_values('unique_id', inplace=True)
    subm_df.sort_values('unique_id', inplace=True)

    gt_df.reset_index(drop=True, inplace=True)
    subm_df.reset_index(drop=True, inplace=True)
    # print(subm_df)
    # print(gt_df)

    # sort gt_df and subm_df by unique_id
    return score(gt_df, subm_df, 'unique_id')

def eval_subm_g10(subm_df, gt_df):

    gt_df["unique_id"] = gt_df["eeg_id"].astype(str) + "_" + gt_df["eeg_sub_id"].astype(str)
    subm_df["unique_id"] = subm_df["eeg_id"].astype(str) + "_" + subm_df["eeg_sub_id"].astype(str)


    g10_ids = gt_df[gt_df["total_votes"] >= 10]["unique_id"].values
    g10_subm = subm_df[subm_df["unique_id"].isin(g10_ids)]
    
    score = eval_subm(g10_subm, gt_df)
    return score

def kl_divergence(solution: pd.DataFrame, submission: pd.DataFrame, epsilon: float, micro_average: bool, sample_weights: Optional[pd.Series]):
    # Overwrite solution for convenience
    for col in solution.columns:
        # Prevent issue with populating int columns with floats
        if not pandas.api.types.is_float_dtype(solution[col]):
            solution[col] = solution[col].astype(float)

        # Clip both the min and max following Kaggle conventions for related metrics like log loss
        # Clipping the max avoids cases where the loss would be infinite or undefined, clipping the min
        # prevents users from playing games with the 20th decimal place of predictions.
        submission[col] = np.clip(submission[col], epsilon, 1 - epsilon)

        y_nonzero_indices = solution[col] != 0
        solution[col] = solution[col].astype(float)
        solution.loc[y_nonzero_indices, col] = solution.loc[y_nonzero_indices, col] * np.log(solution.loc[y_nonzero_indices, col] / submission.loc[y_nonzero_indices, col])
        # Set the loss equal to zero where y_true equals zero following the scipy convention:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html#scipy.special.rel_entr
        solution.loc[~y_nonzero_indices, col] = 0

    if micro_average:
        return np.average(solution.sum(axis=1), weights=sample_weights)
    else:
        return np.average(solution.mean())


def score(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        row_id_column_name: str,
        epsilon: float=10**-15,
        micro_average: bool=True,
        sample_weights_column_name: Optional[str]=None
    ) -> float:
    ''' The Kullbackâ€“Leibler divergence.
    The KL divergence is technically undefined/infinite where the target equals zero.

    This implementation always assigns those cases a score of zero; effectively removing them from consideration.
    The predictions in each row must add to one so any probability assigned to a case where y == 0 reduces
    another prediction where y > 0, so crucially there is an important indirect effect.

    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    solution: pd.DataFrame
    submission: pd.DataFrame
    epsilon: KL divergence is undefined for p=0 or p=1. If epsilon is not null, solution and submission probabilities are clipped to max(eps, min(1 - eps, p).
    row_id_column_name: str
    micro_average: bool. Row-wise average if True, column-wise average if False.

    Examples
    --------
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> score(pd.DataFrame({'id': range(4), 'ham': [0, 1, 1, 0], 'spam': [1, 0, 0, 1]}), pd.DataFrame({'id': range(4), 'ham': [.1, .9, .8, .35], 'spam': [.9, .1, .2, .65]}), row_id_column_name=row_id_column_name)
    0.216161...
    >>> solution = pd.DataFrame({'id': range(3), 'ham': [0, 0.5, 0.5], 'spam': [0.1, 0.5, 0.5], 'other': [0.9, 0, 0]})
    >>> submission = pd.DataFrame({'id': range(3), 'ham': [0, 0.5, 0.5], 'spam': [0.1, 0.5, 0.5], 'other': [0.9, 0, 0]})
    >>> score(solution, submission, 'id')
    0.0
    >>> solution = pd.DataFrame({'id': range(3), 'ham': [0, 0.5, 0.5], 'spam': [0.1, 0.5, 0.5], 'other': [0.9, 0, 0]})
    >>> submission = pd.DataFrame({'id': range(3), 'ham': [0.2, 0.3, 0.5], 'spam': [0.1, 0.5, 0.5], 'other': [0.7, 0.2, 0]})
    >>> score(solution, submission, 'id')
    0.160531...
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    sample_weights = None
    if sample_weights_column_name:
        if sample_weights_column_name not in solution.columns:
            raise ParticipantVisibleError(f'{sample_weights_column_name} not found in solution columns')
        sample_weights = solution.pop(sample_weights_column_name)

    if sample_weights_column_name and not micro_average:
        raise ParticipantVisibleError('Sample weights are only valid if `micro_average` is `True`')

    for col in solution.columns:
        if col not in submission.columns:
            raise ParticipantVisibleError(f'Missing submission column {col}')

    kaggle_metric_utilities.verify_valid_probabilities(solution, 'solution')
    kaggle_metric_utilities.verify_valid_probabilities(submission, 'submission')


    return kaggle_metric_utilities.safe_call_score(kl_divergence, solution, submission, epsilon=epsilon, micro_average=micro_average, sample_weights=sample_weights)