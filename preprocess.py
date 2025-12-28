import os

import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.sparse import vstack, csr_matrix
import scipy.sparse as sp

def split_h5ad(cls_key, adata, test_size=0.2, random_state=2024):
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    train_adata_list, test_adata_list = [], []
    cell_types = adata.obs[cls_key].unique()

    for cell_type in tqdm(cell_types):
        idx = np.where(adata.obs[cls_key] == cell_type)[0]
        if len(idx) < 2:
            train_idx = idx
            test_idx = idx
        else:
            train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state)
        train_adata_list.append(adata[train_idx])
        test_adata_list.append(adata[test_idx])

    train_adata = ad.concat(train_adata_list)
    test_adata = ad.concat(test_adata_list)
    return train_adata, test_adata
