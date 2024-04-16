class HMS_CONFIG:
    DATASET_PATH = "/home/maxc/workspace/kaggle-hms/hms-harmful-brain-activity-classification"
    DATA_PATH = "/home/maxc/workspace/kaggle-hms/data"

TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
SIGNAL_NAME = ['LL','RL','LP','RP','ML']

EEG_FEATS = [['Fp1','F7','T3','T5','O1'],
        ['Fp2','F8','T4','T6','O2'],
        ['Fp1','F3','C3','P3','O1'],
        ['Fp2','F4','C4','P4','O2'],
        ['Fz','Cz','Pz']
        ]
