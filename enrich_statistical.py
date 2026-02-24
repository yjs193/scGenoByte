import json
import scanpy as sc
import pandas as pd
import numpy as np
import gseapy as gp
from scipy.sparse import csr_matrix
import anndata as ad
import matplotlib.pyplot as plt

CELLTYPE_KEY = 'celltype'
ADJ_P_VAL_CUTOFF = 0.05
LOG2FC_CUTOFF = 0.5

if __name__ == '__main__':
    dataset_path = r'/data/js/data/paper_data/raw/Zheng68k/Zheng68K_23856.h5ad'
    adata = sc.read(dataset_path)


    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    gene_set_path = r'../../data/GeneSets.json'
    genesets = json.load(open(gene_set_path))
    geneset_name = "Reactome"
    geneset_dict = genesets[geneset_name]


    print("Starting ssGSEA calculation...")

    input_df = adata.to_df().T


    ssgsea_res = gp.ssgsea(
        data=input_df,
        gene_sets=geneset_dict,
        outdir=None,
        sample_norm_method='rank',
        no_plot=True,
        processes=8
    )


    pathway_names = ssgsea_res.res2d.index.tolist()
    pathways_mtx = ssgsea_res.res2d.T.values.astype(np.float32)

    print(f"ssGSEA completed. Calculated {len(pathway_names)} pathways.")

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


    adata_pathways_filtered = adata_pathways[:, final_specific_pathway_names]


    adata.obsm['pathways_mtx'] = adata_pathways_filtered.X
    adata.uns['pathways_mtx_cols'] = final_specific_pathway_names

    output_path = dataset_path.replace('.h5ad', f'_with_ssGSEA_{geneset_name}.h5ad')
    adata.write_h5ad(output_path)
    print(f"Results saved to {output_path}")