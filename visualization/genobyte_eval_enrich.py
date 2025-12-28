import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random
import scanpy as sc

# Import gseapy
try:
    import gseapy as gp
except ImportError:
    print("!! ERROR: 'gseapy' library not found !!")
    print("Please run: pip install gseapy in your 'pt310' conda environment")
    exit(1)

# Import stats module for p-value calculation
try:
    from scipy import stats
except ImportError:
    print("!! ERROR: 'scipy' library not found !!")
    print("Please run: pip install scipy in your 'pt310' conda environment")
    exit(1)

# ==========================================================
# 0. Setup Plotting Fonts
# ==========================================================
try:
    # Set base font size for ticks, legend, etc.
    plt.rcParams.update({'font.size': 18, 'font.family': 'sans-serif'})
    print("Set base font size to 18 and font family to 'sans-serif'.")
except Exception as e:
    print(f"Warning: Could not set default rcParams. Error: {e}")

# ==========================================================
# 1. (MODIFIED) Configuration
# ==========================================================

# (Input 1) Your *Experimental* h5ad file (PPI-sorted and Padded)
EXPERIMENTAL_ADATA_PATH = r'/data/js/data/paper_data/raw/Zheng68k/Zheng68K_23856.h5ad'

# (Input 2) Your *Control* h5ad file (Original data, for shuffling AND background)
CONTROL_ADATA_PATH = r'/data/js/data/paper_data/raw/Zheng68k/Zheng68K_23856.h5ad'

# (Input 3) Your *Local* Gene Set JSON file
LOCAL_GENESET_JSON_PATH = r'/home/js/bio/PII_cluster/data/GeneSets.json'

# (Input 4) List of Keys to use from the JSON file
GENE_SET_KEYS_TO_ANALYZE = ['Celltype',  'Reactome', 'GO', ]
# GENE_SET_KEYS_TO_ANALYZE = ['GO', ]

# 💥 修改: "Patch" -> "GenoByte"
DEBUG_PATCH_LIMIT = None  # (设为 None 以运行全部 GenoBytes)
PATCH_SIZE = 16
# 💥 修改: "enrichment_comparison_output" -> "genobyte_enrichment_comparison"
OUTPUT_DIR = r'./genobyte_enrichment_comparison'


# ==========================================================
# 2. Helper Functions (MODIFIED)
# ==========================================================

def load_local_gene_sets(json_path, key):
    """
    Loads the specified gene set dictionary from the local GeneSets.json file.
    """
    print(f"\nLoading local gene sets from {json_path} for key: '{key}'...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            all_sets = json.load(f)

        if key not in all_sets:
            print(f"!! CRITICAL ERROR: Key '{key}' not found in {json_path}")
            print(f"    Available keys: {list(all_sets.keys())}")
            return None

        gene_sets_dict = all_sets[key]
        print(f"Successfully loaded {len(gene_sets_dict)} pathways (from '{key}').")

        return gene_sets_dict

    except Exception as e:
        print(f"!! CRITICAL ERROR: Could not load or parse {json_path}. {e}")
        return None


def get_ppi_patches_from_adata(adata_path, patch_size, csv_path=None):
    """
    💥 修改: "Patch" -> "GenoByte"
    Generate GenoBytes directly from h5ad var_names in order (Experimental Group)
    """
    print(f"Loading 'PPI-Padded' (Experimental Group) genes from {adata_path}...")
    try:
        if csv_path is not None:
            print(f"Note: CSV path provided ({csv_path}), but currently not used in this function.")
            gene_list = pd.read_csv(csv_path)['gene'].tolist()
        else:
            adata = sc.read_h5ad(adata_path)
            gene_list = list(adata.var_names)
        print(f"Successfully loaded {len(gene_list)} sorted genes.")
    except Exception as e:
        print(f"!! CRITICAL ERROR: Could not load {adata_path}. {e}")
        return None

    patches = []
    for i in range(0, len(gene_list), patch_size):
        patch = gene_list[i:i + patch_size]
        if len(patch) == patch_size:
            patches.append(patch)
        else:
            # 💥 修改: "patch" -> "GenoByte"
            tqdm.write(f"Note: Discarded incomplete GenoByte at the end of {adata_path} (size: {len(patch)})")

    # 💥 修改: "patches" -> "GenoBytes"
    print(f"Generated {len(patches)} GenoBytes from 'PPI-Padded' h5ad.")
    return patches


def get_random_shuffled_patches(all_genes_list, patch_size, num_patches_to_generate):
    """
    💥 修改: "Patch" -> "GenoByte"
    Load gene list from *original* h5ad, shuffle, and generate GenoBytes (Control Group)
    """
    print(f"Generating 'Random Shuffled' control group for {len(all_genes_list)} genes...")
    shuffled_genes = all_genes_list.copy()
    random.shuffle(shuffled_genes)

    total_genes_needed = num_patches_to_generate * patch_size
    n_orig = len(shuffled_genes)

    final_gene_list = []
    if total_genes_needed > n_orig:
        print(f"Warning: Experimental group gene count ({total_genes_needed}) > Original gene count ({n_orig}).")
        print("Using replacement sampling (random.choices) to fill control group...")
        padding_needed = total_genes_needed - n_orig
        padded_genes = random.choices(shuffled_genes, k=padding_needed)
        final_gene_list = shuffled_genes + padded_genes
    else:
        final_gene_list = shuffled_genes[:total_genes_needed]

    patches = []
    for i in range(0, len(final_gene_list), patch_size):
        patch = final_gene_list[i:i + patch_size]
        if len(patch) == patch_size:  # Ensure patch is full
            patches.append(patch)

    # 💥 修改: "patches" -> "GenoBytes"
    print(f"'Random Shuffled' strategy successfully generated {len(patches)} GenoBytes.")
    return patches


# 💥 移除: get_alphabetical_patches 函数已删除


# ==========================================================
# 3. Core Calculation Function
# ==========================================================

def calculate_enrichment_scores(patch_list, gene_set_dict, background_gene_list, current_key):
    """
    💥 修改: "patches" -> "GenoBytes"
    Core function: Calculate "Functional Enrichment Significance"
    Uses gseapy.enrichr *local mode* and returns -log10(Adj. P-value)
    """
    scores = []
    print(f"\nRunning gseapy.enrichr on {len(patch_list)} GenoBytes (Local Mode, Key: {current_key})...")

    background_set = set(background_gene_list)

    for gene_list in tqdm(patch_list, desc=f"Enriching (Local {current_key})"):
        valid_genes = [
            g for g in gene_list
            if not g.startswith('padded_') and g in background_set
        ]

        if len(valid_genes) < 4:
            scores.append(0.0)
            continue

        try:
            enr_results = gp.enrichr(
                gene_list=valid_genes,
                gene_sets=gene_set_dict,
                background=background_gene_list,
                outdir=None,
                no_plot=True,
                cutoff=1.0
            )

            if enr_results.results.empty:
                scores.append(0.0)
                continue

            min_adj_p_val = enr_results.results['Adjusted P-value'].min()
            epsilon = 1e-300
            score = -np.log10(min_adj_p_val + epsilon)
            scores.append(score)

        except Exception as e:
            tqdm.write(f"Warning: Enrichr failed on a GenoByte: {e}. Skipping...")
            scores.append(0.0)

    return scores


# Helper function for formatting p-values
def format_p_value(p_value):
    """Return significance stars for a given p-value."""
    if p_value < 0.0001:
        return '****'
    elif p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'n.s.'  # (not significant)


# ==========================================================
# 4. (HEAVILY MODIFIED) Main Execution Logic
# ==========================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory ensured: {OUTPUT_DIR}")

    # --- Step 1: Load Background Genes (from Control h5ad) ---
    print(f"Loading 'Original' (Control/Background) genes from {CONTROL_ADATA_PATH}...")
    try:
        adata_orig = sc.read_h5ad(CONTROL_ADATA_PATH)
        original_var_names = list(adata_orig.var_names)
        print(f"Successfully loaded {len(original_var_names)} background genes.")
    except Exception as e:
        print(f"!! CRITICAL ERROR: Could not load {CONTROL_ADATA_PATH}. {e}")
        exit(1)

    # --- Step 2: Get Experimental GenoBytes (Done once) ---
    # 💥 修改: "Patches" -> "GenoBytes"
    ppi_patches_base = get_ppi_patches_from_adata(EXPERIMENTAL_ADATA_PATH, PATCH_SIZE, csv_path=None)
    if ppi_patches_base is None: exit(1)
    n_patches_to_generate = len(ppi_patches_base)
    if n_patches_to_generate == 0:
        print("!! CRITICAL ERROR: Experimental h5ad generated 0 GenoBytes. Terminating.")
        exit(1)

    # --- Step 3a: Get Control GenoBytes (Random) ---
    # 💥 修改: 移除 CALCULATE_RANDOM_GROUP 开关，始终运行
    random_patches_base = get_random_shuffled_patches(
        original_var_names,
        PATCH_SIZE,
        n_patches_to_generate
    )
    if random_patches_base is None: exit(1)

    # --- 💥 Step 3b: (Alphabetical) 已删除 ---

    # --- Step 4: Apply Debug Limit (Done once) ---
    # 💥 修改: "patches" -> "GenoBytes", 移除 "alphabetical"
    if DEBUG_PATCH_LIMIT is not None:
        print(f"\n--- DEBUG: Limiting analysis to {DEBUG_PATCH_LIMIT} GenoBytes ---")
        ppi_patches = ppi_patches_base[:DEBUG_PATCH_LIMIT]
        random_patches = random_patches_base[:DEBUG_PATCH_LIMIT]
        if not ppi_patches or not random_patches:
            print("!! CRITICAL ERROR: Debug limit resulted in zero GenoBytes. Terminating.")
            exit(1)
    else:
        ppi_patches = ppi_patches_base
        random_patches = random_patches_base

    # --- Step 5: Loop, Analyze, and Collect Results ---
    all_results_dfs = []  # List to hold all DataFrames
    stats_results_random = {}  # Dict for PPI vs Random
    # 💥💥 新增: 字典用于存储平均分
    mean_scores_random = {}
    mean_scores_ppi = {}
    # 💥 移除: stats_results_alpha

    print("\n" + "=" * 50)
    print("STARTING BATCH ANALYSIS")
    print(f"Gene Sets to analyze: {GENE_SET_KEYS_TO_ANALYZE}")
    print("=" * 50)

    for gene_set_key in GENE_SET_KEYS_TO_ANALYZE:
        print(f"\n--- Processing: {gene_set_key} ---")

        # 5a. Load the specific gene set
        local_gene_sets = load_local_gene_sets(LOCAL_GENESET_JSON_PATH, gene_set_key)
        if local_gene_sets is None:
            print(f"Skipping {gene_set_key} due to load error.")
            continue

        # 5b. Calculate scores
        print(f"--- Analyzing 'Random Shuffled' (Control 1) (DB: {gene_set_key}) ---")
        random_scores = calculate_enrichment_scores(
            random_patches, local_gene_sets, original_var_names, gene_set_key
        )

        # 💥 移除: alphabetical_scores

        print(f"--- Analyzing 'PPI-Padded' (Experimental Group) (DB: {gene_set_key}) ---")
        ppi_scores = calculate_enrichment_scores(
            ppi_patches, local_gene_sets, original_var_names, gene_set_key
        )

        # 5c. Quantitative Comparison
        mean_random = np.nanmean(random_scores) if random_scores else 0
        mean_ppi = np.nanmean(ppi_scores) if ppi_scores else 0
        metric_label = f"-log10(Min. Adj. P-value) @ {gene_set_key}"

        # 💥💥 新增: 存储平均分
        mean_scores_random[gene_set_key] = mean_random
        mean_scores_ppi[gene_set_key] = mean_ppi

        # Stat 1: PPI vs Random
        p_val_rand = 1.0
        try:
            u_stat_rand, p_val_rand = stats.mannwhitneyu(ppi_scores, random_scores, alternative='greater')
        except ValueError as e:
            print(f"Warning during statistical test (vs Random) for {gene_set_key}: {e}")
            p_val_rand = 1.0
        stats_results_random[gene_set_key] = p_val_rand

        # 💥 移除: Stat 2 (PPI vs Alphabetical)

        # 5d. Print results for this gene set
        print("\n" + "=" * 40)
        print(f"--- Results for: {gene_set_key} ---")
        print(f"    Metric: {metric_label}")
        print(f"    Control Group (Random Shuffled): {mean_random:.6f} (Avg. Score)")
        # 💥 移除: mean_alpha
        print(f"    Experimental Group (PPI-Padded): {mean_ppi:.6f} (Avg. Score)")
        print(f"    Mann-Whitney U (vs Random): p = {p_val_rand:.2e} ({format_p_value(p_val_rand)})")
        # 💥 移除: p_val_alpha
        print("=" * 40)

        # 5e. Add results to the master list
        # 💥 修改: "Patch Type" -> "GenoByte Type"
        df_random = pd.DataFrame(
            {'Score': random_scores, 'GenoByte Type': 'Random Shuffled', 'Gene Set': gene_set_key})
        df_ppi = pd.DataFrame({'Score': ppi_scores, 'GenoByte Type': 'PPI-Padded', 'Gene Set': gene_set_key})

        all_results_dfs.extend([df_random, df_ppi])

    # --- Step 6: Final check and DataFrame concatenation ---
    if not all_results_dfs:
        print("\n!! CRITICAL ERROR: No results were generated from any gene set. Terminating.")
        exit(1)

    print("\nBatch analysis complete. Concatenating all results for plotting...")
    plot_df = pd.concat(all_results_dfs)
    processed_gene_sets = [key for key in GENE_SET_KEYS_TO_ANALYZE if
                           key in stats_results_random]

    # --- Define Palette & Plot Settings ---
    plot_palette = {
        'Random Shuffled': '#a29ff5',  # (Blue-Purple)
        'PPI-Padded': '#9fc7f5'  # (Blue)
    }
    x_hue_order = ['Random Shuffled', 'PPI-Padded']
    # 💥 修改: "Patch" -> "GenoByte"
    plot_title = 'GenoByte Functional Enrichment Comparison'
    plot_filename = 'genobyte_enrichment_boxplot_Random_vs_PPI.svg'
    fig_width = 22  # 2组，宽度设为12

    # =========================================================================
    # --- Step 7: (MODIFIED) Plot 1: Plot all calculated groups together ---
    # =========================================================================
    try:
        print(f"\nGenerating Nature-style grouped boxplot ({' vs '.join(x_hue_order)})...")

        fig, ax = plt.subplots(figsize=(fig_width, 10))
        sns.set_style("ticks")

        # 💥 修改: "Patch Type" -> "GenoByte Type", "plot_d" -> "plot_df"
        sns.boxplot(ax=ax, data=plot_df,  # <--- Corrected this line
                    x='Gene Set', y='Score', hue='GenoByte Type',
                    palette=plot_palette, order=processed_gene_sets, hue_order=x_hue_order,
                    showfliers=False, width=0.56, linewidth=1.6, legend=True)
        # 💥💥💥 END FIX 💥💥💥

        # --- Adjust legend position & FONT ---
        # 💥 修改: fontsize=16
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=False, fontsize=16)

        # 💥 修改: FONT SIZES & "Patch" -> "GenoByte"
        ax.set_title(plot_title, fontsize=22, pad=20)
        ax.set_ylabel(f'Functional Significance Score\n(-log10[Min. Adj. P-value])', fontsize=16, labelpad=15)
        # 💥💥 修改: 'Gene Set Database' -> 'Gene Sets'
        ax.set_xlabel('Gene Sets', fontsize=18, labelpad=15)
        ax.tick_params(axis='both', which='major', labelsize=16, length=6, width=1.5)

        # --- Add SIMPLIFIED statistical annotations ---
        n_hues = len(x_hue_order)
        # --- Add SIMPLIFIED statistical annotations (Optimized for Visuals) ---
        n_hues = len(x_hue_order)
        group_width = 0.8
        box_width = group_width / n_hues
        positions = np.linspace(-group_width / 2 + box_width / 2, group_width / 2 - box_width / 2, n_hues)
        x_offsets = {hue: pos for hue, pos in zip(x_hue_order, positions)}

        # 强制 Y 轴上限
        Y_LIMIT_TOP = 21
        ax.set_ylim(bottom=-0.5, top=Y_LIMIT_TOP)  # 稍微留点底部空间

        for i, gene_set_key in enumerate(processed_gene_sets):
            # 获取当前 Gene Set 的数据子集
            subset = plot_df[plot_df['Gene Set'] == gene_set_key]

            # --- 1. 动态计算每根柱子的视觉高度 (Whisker Top) ---
            # 这样文字就能紧贴着箱线图的须，而不是漂在天上
            tops = {}
            means = {}

            for hue_name in x_hue_order:
                scores = subset[subset['GenoByte Type'] == hue_name]['Score']
                if scores.empty:
                    tops[hue_name] = 0
                    means[hue_name] = 0
                    continue

                # 计算箱线图的视觉上边缘 (Q3 + 1.5IQR)，因为 showfliers=False
                q1 = scores.quantile(0.25)
                q3 = scores.quantile(0.75)
                iqr = q3 - q1
                whisker_top = min(scores.max(), q3 + 1.5 * iqr)

                # 如果数值太小(比如接近0)，给一个最小高度以免文字甚至画在轴下面
                tops[hue_name] = max(whisker_top, 0.5)

                # 获取需要显示的平均分
                if hue_name == 'Random Shuffled':
                    means[hue_name] = mean_scores_random.get(gene_set_key, 0)
                else:
                    means[hue_name] = mean_scores_ppi.get(gene_set_key, 0)

            # --- 2. 绘制数值文字 (紧贴柱子) ---
            x1_rand = i + x_offsets['Random Shuffled']
            x2_ppi = i + x_offsets['PPI-Padded']

            # Random 的文字位置
            y_text_rand = min(tops['Random Shuffled'] + 0.5, Y_LIMIT_TOP - 2)  # 防止顶出去
            ax.text(x1_rand, y_text_rand, f"{means['Random Shuffled']:.2f}",
                    ha='center', va='bottom', fontsize=16, color='black')

            # PPI 的文字位置
            y_text_ppi = min(tops['PPI-Padded'] + 0.5, Y_LIMIT_TOP - 2)
            ax.text(x2_ppi, y_text_ppi, f"{means['PPI-Padded']:.2f}",
                    ha='center', va='bottom', fontsize=16, color='black')

            # --- 3. 绘制显著性横线 (统一画在较高处，但不超过 Y_LIMIT) ---
            # 选取两组中较高的那个位置作为横线的基准，但至少要在 Y=3 以上，不然太低不好看
            # 如果数据被截断了(超过20)，横线就固定画在 19 左右

            visual_max_y = max(y_text_rand, y_text_ppi)
            bar_tips = min(visual_max_y + 1.5, Y_LIMIT_TOP - 1.5)  # 横线高度
            bar_height = bar_tips + 0.5  # 竖线高度

            # 强制横线不低于某个美观高度 (比如 Y=5)，防止Random组太低时横线压着箱子
            bar_tips = max(bar_tips, 5.0)
            bar_height = bar_tips + 0.5

            # 如果接近顶端，固定位置
            if bar_height > Y_LIMIT_TOP - 1:
                bar_height = Y_LIMIT_TOP - 0.5
                bar_tips = bar_height - 0.5

            p_val = stats_results_random.get(gene_set_key, 1.0)
            p_text = format_p_value(p_val)

            # 画线
            ax.plot([x1_rand, x1_rand, x2_ppi, x2_ppi],
                    [bar_tips, bar_height, bar_height, bar_tips],
                    lw=1.5, c='black')

            # 写星号
            ax.text((x1_rand + x2_ppi) / 2, bar_height + 0.1, p_text,
                    ha='center', va='bottom', fontsize=16, color='black')


        # --- Set Y-axis limits more precisely ---
        # 💥 修复: 修正 Y 轴上限 (原 0.7 是个bug)
        # final_y_upper_limit = max_y_for_plot * 1.10  # 增加 10% 顶部空间
        # ax.set_ylim(bottom=min(plot_df['Score'].min() * 0.9, -0.1), top=final_y_upper_limit)
        ax.set_ylim(bottom=min(plot_df['Score'].min() * 0.9, -0.1), top=25)
        sns.despine(ax=ax)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        fig.tight_layout()

        # 💥 修改: "patch" -> "genobyte"
        plot_path = os.path.join(OUTPUT_DIR, plot_filename)

        print(f"\nAttempting to save figure to: {plot_path}")
        # fig.savefig(plot_path, dpi=600, bbox_inches='tight')
        fig.savefig(plot_path,  bbox_inches='tight')
        print("Figure (Combined) saved successfully.")
        plt.close(fig)

    except Exception as e:
        print(f"\n!! CRITICAL ERROR during plotting: {e}")
        import traceback

        traceprint_exc()

    print(f"\n--- Analysis Complete ---")
    if 'plot_path' in locals() and os.path.exists(plot_path):
        print(f"Plot saved to: {plot_path}")
    else:
        print(f"Plot was NOT saved due to an error.")