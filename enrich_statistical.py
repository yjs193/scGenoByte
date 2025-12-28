import json
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import anndata as ad
import matplotlib.pyplot as plt



CELLTYPE_KEY = 'celltype'

ADJ_P_VAL_CUTOFF = 0.05  #
LOG2FC_CUTOFF = 0.5  #
# ----------------------------------------

if __name__ == '__main__':
    dataset_path = r'/data/js/data/paper_data/raw/Zheng68k/Zheng68K_23856.h5ad'
    adata = sc.read(dataset_path)
    adata.X = csr_matrix(adata.X)
    gene_set_path = r'../../data/GeneSets.json'
    genesets = json.load(open(gene_set_path))
    geneset_name =  "Reactome"
    geneset_dict = genesets[geneset_name]  #

    pathway_names = []
    pathway_scores_list = []
    genes_in_data = set(adata.var_names)

    for pathway_name, gene_list in tqdm(geneset_dict.items(), desc="Scoring pathways"):
        genes_in_adata = [gene for gene in gene_list if gene in genes_in_data]
        if len(genes_in_adata) < 5:
            continue
        ctrl_size = min(len(genes_in_adata) - 1, 50)
        if ctrl_size <= 0:
            continue
        temp_score_name = f"temp_score"
        sc.tl.score_genes(
            adata,
            gene_list=genes_in_adata,
            ctrl_size=ctrl_size,
            gene_pool=None,
            n_bins=25,
            score_name=temp_score_name,
            random_state=0
        )
        pathway_names.append(pathway_name)
        pathway_scores_list.append(adata.obs[temp_score_name].values)
        del adata.obs[temp_score_name]



    pathways_mtx = np.stack(pathway_scores_list, axis=1)
    adata_pathways = ad.AnnData(X=pathways_mtx, obs=adata.obs.copy())
    adata_pathways.var_names = pathway_names
    sc.tl.rank_genes_groups(
        adata_pathways,
        groupby=CELLTYPE_KEY,
        method='wilcoxon',
        n_genes=adata_pathways.n_vars  # <-- 关键修改
    )
    result_df = sc.get.rank_genes_groups_df(adata_pathways, group=None)
    filtered_df = result_df[
        (result_df['pvals_adj'] < ADJ_P_VAL_CUTOFF) &
        (result_df['logfoldchanges'] > LOG2FC_CUTOFF)
        ]

    # 4. 提取通路的并集
    top_pathways_set = set(filtered_df['names'])
    final_specific_pathway_names = list(top_pathways_set)



    adata_pathways_filtered = adata_pathways[:, final_specific_pathway_names]

    heatmap_path = dataset_path.replace('.h5ad', f'_pathways_heatmap_{geneset_name}_v3_statistical.png')
    fig = sc.pl.matrixplot(
        adata_pathways_filtered,
        var_names=adata_pathways_filtered.var_names,
        groupby=CELLTYPE_KEY,
        use_raw=False,
        standard_scale='var',  # 按列 Z-score
        title=f"Mean Pathway Score (Z-scaled) by Cell Type\n(p < {ADJ_P_VAL_CUTOFF} & log2fc > {LOG2FC_CUTOFF})",
        show=False,
        return_fig=True
    )

    plt.close()

    pathways_df = pd.DataFrame(
        pathways_mtx,
        index=adata.obs_names,
        columns=pathway_names
    )
    final_pathways_df = pathways_df[final_specific_pathway_names]
    final_pathways_mtx = final_pathways_df.values.astype(np.float32)

    adata.obsm['pathways_mtx'] = final_pathways_mtx
    adata.uns['pathways_mtx_cols'] = final_specific_pathway_names

    csv_path = dataset_path.replace('.h5ad', f'_pathways_matrix_{geneset_name}_v3_statistical.csv')
    final_pathways_df.to_csv(csv_path)
    output_path = dataset_path.replace('.h5ad', f'_with_pathways_{geneset_name}_v3_stat.h5ad')
    adata.write_h5ad(output_path)
