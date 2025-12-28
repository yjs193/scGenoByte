import json
import anndata as ad
import pandas as pd
import scipy.sparse as sparse  # 使用别名方便调用
import numpy as np
from tqdm import tqdm
import scanpy as sc


def pad_and_patch_adata_optimized_v2(adata, patched_gene_list, patch_size=30):
    original_gene_map = {gene: i for i, gene in enumerate(adata.var_names)}
    if not sparse.issparse(adata.X):
        is_sparse = False
        adata_X = adata.X
    else:
        is_sparse = True

        adata_X = adata.X.tocsc()

    new_n_genes = len(patched_gene_list)
    print(f"Original gene count: {len(original_gene_map)}, New padded gene count: {new_n_genes}")


    if is_sparse:
        new_X_lil = sparse.lil_matrix((adata.n_obs, new_n_genes), dtype=adata.X.dtype)
    else:
        new_X_dense = np.zeros((adata.n_obs, new_n_genes), dtype=adata.X.dtype)


    print("Constructing new padded matrix...")
    for i, gene_name in enumerate(tqdm(patched_gene_list, desc="Copying and padding genes", leave=True)):
        if gene_name in original_gene_map:
            orig_idx = original_gene_map[gene_name]
            if is_sparse:
                new_X_lil[:, i] = adata_X[:, orig_idx]
            else:
                new_X_dense[:, i] = adata_X[:, orig_idx]

    # 5. 最终转换和重塑
    if is_sparse:
        print("Converting LIL matrix to CSR...")
        new_X = new_X_lil.tocsr()
    else:
        new_X = new_X_dense

    print("Creating new AnnData object...")
    new_adata = ad.AnnData(
        X=new_X,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=patched_gene_list),
        obsm=adata.obsm.copy(),
        uns=adata.uns.copy()
    )

    print("Padding and patching complete.")
    return new_adata


if __name__ == '__main__':
    ref_adata_file_path = r'/data/js/data/dataset/Zheng68k_ft_padded_ppi_1_16_PPI_alpha_V7.h5ad'
    ref_adata = sc.read(ref_adata_file_path)
    ref_list = ref_adata.var_names.tolist()
    ref_genes_set = set(ref_list)  # 用于计算共同基因
    adata_file_path = "/data/js/data/paper_data/raw/Zheng68k/Zheng68K.h5ad"
    adata = sc.read(adata_file_path)
    adata_genes_set = set(adata.var_names)  # 用于计算共同基因
    common_genes = ref_genes_set.intersection(adata_genes_set)
    num_common_genes = len(common_genes)

    patch_size = 16
    adata = pad_and_patch_adata_optimized_v2(adata, ref_list, patch_size=patch_size)  # 修正：使用 patch_size 变量

    output_path = adata_file_path.replace('.h5ad', '_23856.h5ad')
    print(f"Saving patched AnnData to {output_path} ...")

    adata.write(output_path)
    print("处理完成。")