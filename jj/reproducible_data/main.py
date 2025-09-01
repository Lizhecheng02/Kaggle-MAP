import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def create_folds(TRAIN_PATH:str, OUTPUT_DIR_PATH:str, OUTPUT_FILE_NAME:str, k:int = 5, random_state:int = 42):

    # Ensure that the training data exists
    assert os.path.exists(TRAIN_PATH), f"Error: {TRAIN_PATH} does not exist."

    # Create the output directory if it does not exist
    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
    
    # Load training data
    trn = pd.read_csv(TRAIN_PATH)
    trn['Misconception'] = trn['Misconception'].fillna("NA")
    trn['QuestionId'] = trn['QuestionId'].astype(str)
    trn['label'] = trn["Category"] + ":" + trn['Misconception']
    trn = trn.drop_duplicates().reset_index(drop=True)

    # Create folds
    skf = StratifiedKFold(n_splits=k, random_state=random_state, shuffle=True)

    trn['fold'] = 0
    for fold, (train_idx, valid_idx) in enumerate(skf.split(trn, trn['QuestionId'] + ":" + trn['label'])):
        trn.loc[valid_idx, 'fold'] = fold

    print(trn)

    trn.to_parquet(f"{OUTPUT_DIR_PATH}/{OUTPUT_FILE_NAME}")


if __name__ == "__main__":

    create_folds("../data/train.csv", "../outputs", "train_fold.parquet", k=5, random_state=42)