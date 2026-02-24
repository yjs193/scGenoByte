import json
import scanpy as sc
import pandas as pd
import numpy as np
import gseapy as gp  # 关键新增：导入 gseapy
from scipy.sparse import csr_matrix
import anndata as ad
import matplotlib.pyplot as plt

CELLTYPE_KEY = 'celltype'
ADJ_P_VAL_CUTOFF = 0.05
LOG2FC_CUTOFF = 0.5

if __name__ == '__main__':
    dataset_path = r'/data/js/data/paper_data/raw/Zheng68k/Zheng68K_23856.h5ad'
    adata = sc.read(dataset_path)

    # 确保数据已标准化 (ssGSEA 建议在 Normalized 数据上运行)
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    gene_set_path = r'../../data/GeneSets.json'
    genesets = json.load(open(gene_set_path))
    geneset_name = "Reactome"
    geneset_dict = genesets[geneset_name]

    # --- 核心修改部分：使用 gseapy.ssgsea 替代循环打分 ---
    print("Starting ssGSEA calculation...")

    # gseapy 要求输入为 DataFrame，行是基因，列是细胞
    input_df = adata.to_df().T

    # 运行 ssGSEA
    # sample_norm_method='rank' 是 ssGSEA 的核心逻辑
    ssgsea_res = gp.ssgsea(
        data=input_df,
        gene_sets=geneset_dict,
        outdir=None,  # 不保存到磁盘，直接拿内存结果
        sample_norm_method='rank',
        no_plot=True,
        processes=8  # 根据您的 CPU 核心数调整
    )

    # 提取结果矩阵 (Cell x Pathway)
    # ssgsea_res.res2d 的行是通路，列是细胞，需要转置
    pathway_names = ssgsea_res.res2d.index.tolist()
    pathways_mtx = ssgsea_res.res2d.T.values.astype(np.float32)

    print(f"ssGSEA completed. Calculated {len(pathway_names)} pathways.")

    # --- 后续统计筛选逻辑 (保持不变，但操作对象变为 ssGSEA 分数) ---
    adata_pathways = ad.AnnData(X=pathways_mtx, obs=adata.obs.copy())
    adata_pathways.var_names = pathway_names

    sc.tl.rank_genes_groups(
        adata_pathways,
        groupby=CELLTYPE_KEY,
        method='wilcoxon',
        n_genes=adata_pathways.n_vars
    )

    result_df = sc.get.rank_genes_groups_df(adata_pathways, group=None)
    filtered_df = result_df[
        (result_df['pvals_adj'] < ADJ_P_VAL_CUTOFF) &
        (result_df['logfoldchanges'] > LOG2FC_CUTOFF)
        ]

    top_pathways_set = set(filtered_df['names'])
    final_specific_pathway_names = list(top_pathways_set)

    # 保存与可视化
    adata_pathways_filtered = adata_pathways[:, final_specific_pathway_names]

    # 保存结果
    adata.obsm['pathways_mtx'] = adata_pathways_filtered.X
    adata.uns['pathways_mtx_cols'] = final_specific_pathway_names

    output_path = dataset_path.replace('.h5ad', f'_with_ssGSEA_{geneset_name}.h5ad')
    adata.write_h5ad(output_path)
    print(f"Results saved to {output_path}")