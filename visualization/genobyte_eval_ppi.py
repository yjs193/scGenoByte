import matplotlib
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random
import itertools
from scipy import stats


try:
    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'sans-serif',
        'axes.unicode_minus': False
    })
except Exception as e:
    print(f"Warning: Could not set default rcParams. Error: {e}")


ORIGINAL_ADATA_PATH = r'/data/js/data/paper_data/raw/Zheng68k/Zheng68K_23856.h5ad'
LOCAL_PPI_LINKS_CSV = r'/data/js/data/format_h_sapiens.csv'

CONFIDENCE_THRESHOLD = 0.4
PATCH_SIZE = 16
OUTPUT_DIR = r'/data/js/res/scGET'  # 版本号 v3


def load_local_ppi_graph(links_path, confidence_threshold=0.4):
    print(f"--- Loading PPI Graph (> {confidence_threshold}) ---")
    try:
        links_df = pd.read_csv(links_path, header=0, names=['index', 'g1_symbol', 'g2_symbol', 'conn'])
    except FileNotFoundError:
        return None
    links_df_high_conf = links_df[links_df['conn'] >= confidence_threshold]
    interaction_graph = set()
    for _, row in tqdm(links_df_high_conf.iterrows(), total=len(links_df_high_conf), desc="Building graph"):
        p1, p2 = sorted([row['g1_symbol'], row['g2_symbol']])
        interaction_graph.add((p1, p2))
    return interaction_graph


def get_ordered_patches(all_genes_list, patch_size=16):
    patches = []
    for i in range(0, len(all_genes_list), patch_size):
        patch = all_genes_list[i:i + patch_size]
        if len(patch) == patch_size: patches.append(patch)
    return patches


def get_random_shuffled_patches(all_genes_list, patch_size=16, num_patches_to_generate=0):
    shuffled_genes = all_genes_list.copy()
    random.shuffle(shuffled_genes)
    total_genes_needed = num_patches_to_generate * patch_size
    if len(shuffled_genes) < total_genes_needed:
        padding = random.choices(shuffled_genes, k=total_genes_needed - len(shuffled_genes))
        shuffled_genes += padding
    else:
        shuffled_genes = shuffled_genes[:total_genes_needed]
    patches = []
    for i in range(0, len(shuffled_genes), patch_size):
        if len(shuffled_genes[i:i + patch_size]) == patch_size: patches.append(shuffled_genes[i:i + patch_size])
    return patches


def calculate_interaction_density(patch_list, interaction_graph):
    scores = []
    for gene_list in tqdm(patch_list, desc="Calc Density"):
        n = len(gene_list)
        if n < 2:
            scores.append(0.0)
            continue
        max_possible_edges = (n * (n - 1)) / 2
        actual_edges = 0
        for gene_a, gene_b in itertools.combinations(gene_list, 2):
            p1, p2 = sorted([gene_a, gene_b])
            if (p1, p2) in interaction_graph: actual_edges += 1
        density = actual_edges / max_possible_edges if max_possible_edges > 0 else 0.0
        scores.append(density)
    return scores


def format_p_value(p_value):
    if p_value < 0.0001:
        return '**** (p < 1e-4)'
    elif p_value < 0.001:
        return '*** (p < 1e-3)'
    elif p_value < 0.01:
        return '** (p < 0.01)'
    else:
        return f'n.s. (p={p_value:.2f})'


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    interaction_graph = load_local_ppi_graph(LOCAL_PPI_LINKS_CSV, CONFIDENCE_THRESHOLD)
    if interaction_graph is None: exit(1)

    adata_orig = sc.read_h5ad(ORIGINAL_ADATA_PATH, backed='r')
    original_var_names = list(adata_orig.var_names)

    ordered_patches = get_ordered_patches(original_var_names, PATCH_SIZE)
    random_patches = get_random_shuffled_patches(original_var_names, PATCH_SIZE, len(ordered_patches))

    print("计算分数...")
    random_scores = calculate_interaction_density(random_patches, interaction_graph)
    ordered_scores = calculate_interaction_density(ordered_patches, interaction_graph)

    # --- 统计 ---
    u_stat, p_val = stats.mannwhitneyu(ordered_scores, random_scores, alternative='greater')
    mean_random = np.mean(random_scores)
    mean_ordered = np.mean(ordered_scores)

    # --- 绘图数据准备 ---
    df_random = pd.DataFrame({'Score': random_scores, 'Strategy': 'Random Shuffled'})
    df_ordered = pd.DataFrame({'Score': ordered_scores, 'Strategy': 'PPI-Guided Partitioning'})
    plot_df = pd.concat([df_random, df_ordered])

    c_random = '#a29ff5'
    c_ppi = '#5ca4f5'
    palette = {'Random Shuffled': c_random, 'PPI-Guided Partitioning': c_ppi}

    # 1. 自动检测 Y 轴限制
    kde_ppi = stats.gaussian_kde(ordered_scores)
    x_grid = np.linspace(0, max(ordered_scores), 200)
    ppi_density_max = max(kde_ppi(x_grid))
    y_limit_zoomed = ppi_density_max * 1.3

    fig, ax = plt.subplots(figsize=(10, 7))

    # 1. 先画紫色 (Random)
    sns.kdeplot(
        data=df_random, x='Score', color=c_random, label='Random Shuffled',
        fill=True, alpha=0.4, linewidth=2.5, cut=0, ax=ax
    )

    # 2. 后画蓝色 (PPI) - 这样蓝色会叠在紫色上面（如果重叠的话），或者您可以交换顺序
    sns.kdeplot(
        data=df_ordered, x='Score', color=c_ppi, label='PPI-Guided Partitioning',
        fill=True, alpha=0.4, linewidth=2.5, cut=0, ax=ax
    )

    ax.set_ylim(0, y_limit_zoomed)

    # 均值线
    ax.axvline(x=mean_ordered, color=c_ppi, linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(x=mean_random, color=c_random, linestyle='--', linewidth=2, alpha=0.8)


    text_y_pos = y_limit_zoomed * 0.65
    ax.text(0.03, text_y_pos,
            f"Random Peak \ncontinues to ~600 \u2191\n(Mean: {mean_random:.4f})",
            color=c_random, fontsize=12, ha='left', va='top')

    ax.text(mean_ordered + 0.01, y_limit_zoomed * 0.8,
            f"PPI Mean: {mean_ordered:.4f}",
            color=c_ppi, fontsize=12, ha='left', va='bottom')


    ax.legend(loc='lower right', bbox_to_anchor=(0.98, 0.10), title='Strategy', title_fontsize=14, fontsize=14)

    # 标题与标签
    ax.set_title(f'Distribution of GenoByte Density with Conn > {CONFIDENCE_THRESHOLD}\n(Zoomed View)', fontsize=24,
                 pad=20)
    ax.set_xlabel('Interconnection Density Score', fontsize=20)
    ax.set_ylabel('Density (Zoomed)', fontsize=20)

    # 统计结果
    stats_text = f"Mann-Whitney U: {format_p_value(p_val)}"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, ha='right', va='top', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))


    ax_ins = ax.inset_axes([0.55, 0.45, 0.4, 0.4])

    sns.kdeplot(
        data=df_random, x='Score', color=c_random,
        fill=True, alpha=0.6, linewidth=2, cut=0, ax=ax_ins
    )
    sns.kdeplot(
        data=df_ordered, x='Score', color=c_ppi,
        fill=True, alpha=0.6, linewidth=2, cut=0, ax=ax_ins
    )

    ax_ins.set_title('Full Scale View', fontsize=12)
    ax_ins.set_xlabel('')
    ax_ins.set_ylabel('')
    ax_ins.tick_params(axis='both', which='major', labelsize=10)
    ax_ins.grid(True, linestyle=':', alpha=0.5)

    sns.despine(ax=ax)

    save_path = os.path.join(OUTPUT_DIR, f'density_zoomed_legend_fixed_{CONFIDENCE_THRESHOLD}.svg')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"可视化完成，图片已保存至: {save_path}")