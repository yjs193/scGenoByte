@torch.no_grad()
def evaluate_and_visualize_patch_embeddings(data_loader, model, device, patch_protein_targets, output_dir,
                                            model_name_tag):
    """
    Performs Representational Similarity Analysis (RSA) and UMAP visualization.
    IMPROVEMENTS:
    1. RSA: Applies Hierarchical Clustering reordering to show structure (blocks) instead of noise.
    2. UMAP: Applies K-Means clustering on UMAP coords to colorize distinct clusters.
    """
    print(f"[{model_name_tag}] Running patch embedding analysis...")
    model.eval()

    all_pred_patch_embeds = []
    total_patch_sum = None
    total_samples = 0
    for batch in tqdm(data_loader):
        images = batch[0].to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            _, patch_embed = model.module.forward_features(images)
        batch_sum = patch_embed.sum(dim=0)

        if total_patch_sum is None:
            total_patch_sum = batch_sum
        else:
            total_patch_sum += batch_sum

        total_samples += patch_embed.shape[0]

    avg_pred_patch_embeds = (total_patch_sum / total_samples).cpu().numpy()

    print(f" Computed average embeddings for {total_samples} cells.")

    truth_patch_embeds = patch_protein_targets.cpu().numpy()

    print("  Calculating RSA (Similarity Matrices)...")
    sim_pred = cosine_similarity(avg_pred_patch_embeds)
    sim_truth = cosine_similarity(truth_patch_embeds)

    triu_indices = np.triu_indices_from(sim_pred, k=1)
    rsa_score = np.corrcoef(sim_pred[triu_indices], sim_truth[triu_indices])[0, 1]
    print(f"  [{model_name_tag}] RSA Score: {rsa_score:.4f}")

    N_PATCHES_TO_PLOT = 2000  # 增加数量以看到更多结构，如果太慢可改回 1000
    DPI_SETTING = 700

    num_available = sim_pred.shape[0]
    if num_available < N_PATCHES_TO_PLOT:
        N_PATCHES_TO_PLOT = num_available

    print(f"  Preparing Aesthetic RSA Heatmap (first {N_PATCHES_TO_PLOT} patches)...")


    truth_subset = truth_patch_embeds[:N_PATCHES_TO_PLOT]
    pred_subset = avg_pred_patch_embeds[:N_PATCHES_TO_PLOT]

    print("  > Computing Hierarchical Clustering order based on Protein Embeddings...")
    linkage_matrix = sch.linkage(truth_subset, method='ward', metric='euclidean')
    dendro = sch.dendrogram(linkage_matrix, no_plot=True)
    reorder_idx = dendro['leaves']  # 获取排序后的索引

    sim_truth_viz = cosine_similarity(truth_subset)[reorder_idx, :][:, reorder_idx]
    sim_pred_viz = cosine_similarity(pred_subset)[reorder_idx, :][:, reorder_idx]

    def normalize_matrix(matrix):
        m_min = matrix.min()
        m_max = matrix.max()
        return (matrix - m_min) / (m_max - m_min + 1e-8)

    sim_truth_viz = normalize_matrix(sim_truth_viz)
    sim_pred_viz = normalize_matrix(sim_pred_viz)
    fig_rsa, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 11))

    sns.heatmap(sim_pred_viz, ax=ax1, cmap='RdBu_r', cbar=True, square=True, vmin=-0.2, vmax=1)
    ax1.set_title(f"Model GenoByte Similarity (Reordered)", fontsize=24)
    ax1.set_xlabel("Reordered Index", fontsize=20)
    ax1.set_xticks([])
    ax1.set_yticks([])

    sns.heatmap(sim_truth_viz, ax=ax2, cmap='RdBu_r', cbar=True, square=True, vmin=-0.2, vmax=1)
    ax2.set_title(f"Ground Truth Protein Similarity (Reordered)", fontsize=24)
    ax2.set_xlabel("Reordered Index", fontsize=20)
    ax2.set_xticks([]);
    ax2.set_yticks([])

    fig_rsa.suptitle(f"Structural Consistency (RSA: {rsa_score:.4f})\nOrdered by Protein Structure Clustering",
                     fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path_rsa = os.path.join(output_dir, f"compare_rsa_clustered_{model_name_tag}.png")
    plt.savefig(save_path_rsa, dpi=1000, bbox_inches='tight',format='png')
    print(f"  RSA heatmap saved to: {save_path_rsa}")
    plt.close(fig_rsa)

    print(f"  Calculating UMAP & Coloring Clusters...")
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    reducer = umap.UMAP(
        n_neighbors=30,  # 原为50。降低此值关注局部，使簇更紧凑
        min_dist=0.2,  # 原为0.1。设为0允许点重叠，视觉上更紧密
        n_components=2,
        random_state=42
    )

    umap_pred = reducer.fit_transform(pred_subset)
    umap_truth = reducer.fit_transform(truth_subset)

    n_clusters_viz = 2

    kmeans_truth = KMeans(n_clusters=n_clusters_viz, random_state=42, n_init=10).fit(umap_truth)
    labels_truth = kmeans_truth.labels_
    labels_pred = labels_truth

    fig_umap, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    scatter1 = ax1.scatter(umap_pred[:, 0], umap_pred[:, 1],
                           c=labels_pred, cmap='Spectral',  # 使用 'Spectral', 'tab10', 'jet' 等彩色 cmap
                           s=18, alpha=0.8)
    ax1.set_title(f"Model Patch Space (Clustered)", fontsize=22)

    from matplotlib.ticker import NullLocator
    ax1.set_xticks([])
    ax1.set_yticks([])

    scatter2 = ax2.scatter(umap_truth[:, 0], umap_truth[:, 1],
                           c=labels_truth, cmap='Spectral',
                           s=18, alpha=0.8)
    ax2.set_title(f"Protein Patch Space (Clustered)", fontsize=22)

    ax2.set_xticks([]);
    ax2.set_yticks([])
    fig_umap.suptitle(f"Embedding Space Structure Comparison", fontsize=24)
    plt.tight_layout()

    save_path_umap = os.path.join(output_dir, f"compare_umap_colored_{model_name_tag}.svg")
    plt.savefig(save_path_umap)
    print(f"  UMAP plot saved to: {save_path_umap}")
    plt.close(fig_umap)

    return rsa_score