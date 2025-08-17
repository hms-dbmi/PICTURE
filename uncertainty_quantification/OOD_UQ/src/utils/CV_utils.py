from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import pandas as pd
from torch.utils.data import Subset

## GroupShuffleSplit with validation set



def group_random_split(
    dataset,
    lengths=[0.5,0.3,0.2],
    seed=42):
    pids = [x['patient_id'] for x in dataset.dict_path_labels]
    # convert list of IDs to array of ints
    # pids = np.array(pids)

    pids = pd.Series(pids)
    # pids, unique_strings = pd.factorize(pids)
    if len(lengths) == 3:
        #contains train, val, test
        use_val=True
        test_size = lengths[2]
        train_size = lengths[0]
        val_size = lengths[1]
        if isinstance(test_size,float):
            val_size = val_size/(1-test_size)
        gss_outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        gss_inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        idx = np.arange(len(pids))
        for train_val_idx, test_idx in gss_outer.split(idx, groups=pids):
            train_val_pids = pids[train_val_idx]
            train_val_idx = idx[train_val_idx]

            for train_idx, val_idx in gss_inner.split(train_val_idx, groups=train_val_pids):
                train_idx = train_val_idx[train_idx]
                val_idx = train_val_idx[val_idx]
                pid_train = pd.unique(pids[train_idx])
                pid_val = pd.unique(pids[val_idx])
                pid_test = pd.unique(pids[test_idx])
                print(f"train: {len(pid_train)} patients, val: {len(pid_val)} patients, test: {len(pid_test)} patients, total: {len(pid_train)+len(pid_val)+len(pid_test)} patients")
                overlap_train_val = np.intersect1d(pid_train,pid_val)
                overlap_train_test = set(pid_train).intersection(set(pid_test))
                assert len(overlap_train_val) == 0, f"train and val overlap: {overlap_train_val}"
                assert len(overlap_train_test) == 0, f"train and test overlap: {overlap_train_test}"



                yield Subset(dataset,train_idx), Subset(dataset,val_idx), Subset(dataset,test_idx)
                # yield train_idx, val_idx, test_idx
    else:
        use_val=False
        test_size = lengths[1]
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        idx = np.arange(len(pids))
        for train_idx, test_idx in gss.split(idx, groups=pids):
            yield  Subset(dataset,train_idx), Subset(dataset,test_idx)

    
