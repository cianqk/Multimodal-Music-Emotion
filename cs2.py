import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

df = pd.read_csv(r"G:\ppppp\KongZhi\.venv\music_dataset\metadata.csv", encoding='utf-8-sig')
df = df.loc[:, ~df.columns.str.startswith('Unnamed')].reset_index(drop=True)

def stratified_split_three_way(df, test_size=0.15, val_size=0.15, random_state=42):
    stratify_col = df['theme_dominant'].astype(str)
    counts = stratify_col.value_counts()
    rare_labels = counts[counts < 2].index
    stratify_col = stratify_col.apply(lambda x: 'rare' if x in rare_labels else x)

    if (stratify_col.value_counts() < 2).any():
        idx = np.arange(len(df))
        train_val_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size/(1-test_size), random_state=random_state)
        return train_idx, val_idx, test_idx

    idx = np.arange(len(df))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_val_idx, test_idx in sss1.split(idx, stratify_col):
        pass

    stratify_col_train_val = stratify_col.iloc[train_val_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=random_state)
    for sub_train_idx, sub_val_idx in sss2.split(train_val_idx, stratify_col_train_val):
        pass

    train_idx = train_val_idx[sub_train_idx]
    val_idx = train_val_idx[sub_val_idx]
    return train_idx, val_idx, test_idx

for seed in [30, 48, 52]:
    train_idx, val_idx, test_idx = stratified_split_three_way(df, random_state=seed)
    s_train, s_val, s_test = set(train_idx), set(val_idx), set(test_idx)
    print(f"\n种子 {seed}")
    print(f"  train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    print(f"  train ∩ test: {len(s_train & s_test)}")
    print(f"  train ∩ val: {len(s_train & s_val)}")
    print(f"  val ∩ test: {len(s_val & s_test)}")
    print(f"  并集大小: {len(s_train | s_val | s_test)} （应为 {len(df)}）")
    print(f"  train 范围: {min(train_idx)}~{max(train_idx)}")