import pandas
import numpy as np
from sklearn.model_selection import GroupShuffleSplit,StratifiedGroupKFold
import pandas as pd

def get_group_stratified_kfold_split(df,seed=123,n_splits=1,split_val=True,group_id='slide'):
    ## Custom stratified KFold split with validation data
    id = df[group_id].to_numpy()
    label = df['label'].to_numpy().astype(np.int64)
    pid, count = np.unique(id,return_counts=True)

    # pid = np.unique(id)
    # label = np.unique(label)
    idx = np.array(range(df.shape[0]))
    ## split target & nontarget separately, grouped by patient
    skf = StratifiedGroupKFold(n_splits=n_splits, random_state=seed, shuffle=True)

    if split_val:
        idx_train_CV = [np.array([])] * n_splits
        idx_val_CV =  [np.array([])] * n_splits
        idx_test_CV =  [np.array([])] * n_splits
        
        for i, (train_index, test_index) in enumerate(skf.split(idx, label, id)):
            for j in range(n_splits):
                if j == i:
                    idx_test_CV[j] = np.append(idx_test_CV[j],idx[test_index])
                elif (j - i) % n_splits == 1:
                    idx_val_CV[j] = np.append(idx_val_CV[j],idx[test_index])
                else:
                    idx_train_CV[j] = np.append(idx_train_CV[j],idx[test_index])

        for i in range(len(idx_train_CV)):
            idx_train = idx_train_CV[i]
            idx_val = idx_val_CV[i]
            idx_test = idx_test_CV[i]
            #yield idx_train, X_test, Y_train, Y_test
            yield i, df.loc[idx_train].reset_index(), df.loc[idx_val].reset_index(), df.loc[idx_test].reset_index()
    else:
        for i, (train_index, test_index) in enumerate(skf.split(idx, label, id)):
            yield i, df.loc[train_index].reset_index(),df.loc[test_index].reset_index()


def get_group_stratified_shuffle_split(df,seed=123,n_splits=1,train_size=0.5,val_size=None,group_id='slide'):
    ##implementing group stratified shuffle split
    id = df[group_id].to_numpy()
    label = df['label'].to_numpy().astype(np.int64)

    df0 = df.loc[label==0].reset_index(drop=True)
    df1 = df.loc[label==1].reset_index(drop=True)
    id0 = df0[group_id].to_numpy()
    id1 = df1[group_id].to_numpy()
    idx0 = np.array(range(df0.shape[0]))
    idx1 = np.array(range(df1.shape[0]))

    pid0, count0 = np.unique(id0,return_counts=True)
    pid1, count1 = np.unique(id1,return_counts=True)

    test_size= 1 - train_size
    if val_size is None:
        test_size= test_size - val_size

    gs0 = GroupShuffleSplit(n_splits=n_splits,test_size=test_size, random_state=seed)
    gs1 = GroupShuffleSplit(n_splits=n_splits,test_size=test_size, random_state=seed)
    iter0 = gs0.split(idx0, None, id0)
    iter1 = gs1.split(idx1, None, id1)
    if val_size is None:
        for i in range(n_splits):
            train_index0, test_index0 = next(iter0)
            train_index1, test_index1 = next(iter1)
            # for i, (train_index, test_index) in enumerate(gs.split(idx, label, id)):
            df_train = pd.concat([
                df0.loc[train_index0],
                df1.loc[train_index1],
            ]).reset_index(drop=True)
            df_test = pd.concat([
                df0.loc[test_index0],
                df1.loc[test_index1],
            ]).reset_index(drop=True)
            yield i, df_train, df_test
    else:
        train_size_inner = train_size / (train_size +val_size)
        for i in range(n_splits):
            train_index0, test_index0 = next(iter0)
            train_index1, test_index1 = next(iter1)

            gs0_inner = GroupShuffleSplit(n_splits=1,train_size=train_size_inner, random_state=seed)
            gs1_inner = GroupShuffleSplit(n_splits=1,train_size=train_size_inner, random_state=seed)
            iter0_inner = gs0.split(train_index0, None, id0[train_index0])
            iter1_inner = gs1.split(train_index1, None, id0[train_index1])

            train_index0, val_index0 = next(iter0_inner)
            train_index1, val_index1 = next(iter1_inner)

            df_train = pd.concat([
                df0.loc[train_index0],
                df1.loc[train_index1],
            ]).reset_index(drop=True)
            df_val = pd.concat([
                df0.loc[val_index0],
                df1.loc[val_index1],
            ]).reset_index(drop=True)
            df_test = pd.concat([
                df0.loc[test_index0],
                df1.loc[test_index1],
            ]).reset_index(drop=True)
            yield i, df_train, df_val, df_test
