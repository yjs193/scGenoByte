import pandas as pd
import networkx as nx
import leidenalg as la
import igraph as ig
import plotly.graph_objects as go
import plotly.express as px  # UMAP 需要
import os
import scanpy as sc
import random
import numpy as np
from collections import defaultdict
import json
import umap.umap_ as umap
import scipy.sparse.csgraph

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

DEBUG_MODE = False
NUM_EDGES_FOR_DEBUG = 50000

ALPHA_HOMOLOGY_WEIGHT = 0.2  # 您可以根据需要调整这个系数

PPI_FILE_PATH = r'/data/js/data/format_h_sapiens.csv'
H5AD_FILE_PATH = r"/data/js/data/dataset/Baron_ft_preprocessed_data.h5ad"
HOMOLOGY_FILE_PATH = r'/home/js/bio/PII_cluster/data/HOMOLOGY_FILE.txt'

OUTPUT_DIR = './output/community_hybrid_homology'
OUTPUT_VIZ_DIR = './output/community_visuals_hybrid_homology'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(OUTPUT_VIZ_DIR):
    os.makedirs(OUTPUT_VIZ_DIR)


def load_homology_scores(file_path):
    print(f"Load: {file_path}...")
    try:
        df = pd.read_csv(file_path, sep='\t')

        if df.shape[1] < 6:
            return {}

        gene1_col = df.columns[0]
        gene2_col = df.columns[1]
        score_col1 = df.columns[4]
        score_col2 = df.columns[5]


        homology_dict = {}
        for _, row in df.iterrows():
            g1 = str(row[gene1_col]).strip()
            g2 = str(row[gene2_col]).strip()

            def clean_score(val):
                if pd.isna(val): return 0.0
                try:
                    return float(str(val).replace('%', '').strip())
                except ValueError:
                    return 0.0

            s1 = clean_score(row[score_col1])
            s2 = clean_score(row[score_col2])

            avg_score = (s1 + s2) / 2.0

            key = tuple(sorted((g1, g2)))

            if key in homology_dict:
                homology_dict[key] = max(homology_dict[key], avg_score)
            else:
                homology_dict[key] = avg_score

        print(f"加载了 {len(homology_dict)} 对同源关系。")
        return homology_dict

    except Exception as e:
        print(f"加载同源文件失败: {e}")
        return {}


homology_scores = load_homology_scores(HOMOLOGY_FILE_PATH)

try:
    df_full = pd.read_csv(PPI_FILE_PATH, header=0, index_col=0)
    df_full.columns = [col.strip() for col in df_full.columns]
    print("PPI网络数据加载成功，总共 {} 条连接。".format(len(df_full)))
except FileNotFoundError:
    print(f"Error '{PPI_FILE_PATH}'。")
    exit()

try:
    gene_list = sc.read(H5AD_FILE_PATH).var_names.tolist()
except FileNotFoundError:
    print("Error")
    exit()

original_count = len(df_full)
gene_set = set(gene_list)
df_filtered = df_full[df_full['g1_symbol'].isin(gene_set) | df_full['g2_symbol'].isin(gene_set)]
filtered_count = len(df_filtered)

df_subset = df_filtered

if DEBUG_MODE:
    if filtered_count > NUM_EDGES_FOR_DEBUG:
        df_subset = df_subset.head(NUM_EDGES_FOR_DEBUG)
        print(f"{NUM_EDGES_FOR_DEBUG}")
    else:
        print(f"{filtered_count}")

print(f"Alpha_Homology = {ALPHA_HOMOLOGY_WEIGHT})...")
G = nx.Graph()
num_edges_updated = 0
print("(PPI + Homology)...")

# 预先处理 PPI 数据以提高效率
for idx, row in df_subset.iterrows():
    u = row['g1_symbol']
    v = row['g2_symbol']

    try:
        ppi_score = float(row['conn'])
    except (ValueError, TypeError):
        ppi_score = 0.1

    key = tuple(sorted((u, v)))
    homology_score = homology_scores.get(key, 0.0)
    final_weight = ppi_score + (homology_score * ALPHA_HOMOLOGY_WEIGHT)

    G.add_edge(u, v, weight=final_weight)

    if homology_score > 0:
        num_edges_updated += 1

print(f"\n混合权重计算完成。{num_edges_updated} / {G.number_of_edges()} 条边因同源分数 > 0 而增加了权重。")
print(f"网络构建完成。包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边。")

if G.number_of_nodes() == 0:
    exit("Error")


def iterative_leiden_partitioning(graph, min_size=10, max_size=50, initial_resolution=1.0, resolution_increment=0.5):
    final_partition = {}
    final_cluster_id = 0
    graphs_to_process = [(graph, initial_resolution)]
    iteration_step = 0

    while graphs_to_process:
        iteration_step += 1
        current_graph, current_resolution = graphs_to_process.pop(0)

        if current_graph.number_of_nodes() < min_size:
            continue

        G_ig = ig.Graph.from_networkx(current_graph)
        if 'weight' not in G_ig.es.attributes():
            G_ig.es['weight'] = [1.0] * G_ig.ecount()

        partition = la.find_partition(G_ig, la.RBConfigurationVertexPartition,
                                      weights='weight',  # 使用我们计算的混合权重
                                      resolution_parameter=current_resolution,
                                      seed=42)
        clusters = defaultdict(list)
        for i, membership_id in enumerate(partition.membership):
            node_name = G_ig.vs[i]['_nx_name']
            clusters[membership_id].append(node_name)

        if len(clusters) == 1:
            nodes_in_single_cluster = list(clusters.values())[0]
            cluster_size = len(nodes_in_single_cluster)
            if cluster_size > max_size:
                new_resolution = current_resolution + resolution_increment
                graphs_to_process.append((current_graph, new_resolution))
            else:
                for node in nodes_in_single_cluster:
                    final_partition[node] = final_cluster_id
                final_cluster_id += 1
            continue

        for cid, nodes in clusters.items():
            cluster_size = len(nodes)
            if cluster_size > max_size:
                subgraph = current_graph.subgraph(nodes)
                graphs_to_process.append((subgraph, 1.0))
            else:
                if cluster_size >= min_size:
                    for node in nodes:
                        final_partition[node] = final_cluster_id
                    final_cluster_id += 1
                else:
                    pass

    return final_partition


MIN_CLUSTER_SIZE = 1
MAX_CLUSTER_SIZE = 16
INITIAL_RESOLUTION = 1.0

final_partition_dict = iterative_leiden_partitioning(G,
                                                     min_size=MIN_CLUSTER_SIZE,
                                                     max_size=MAX_CLUSTER_SIZE,
                                                     initial_resolution=INITIAL_RESOLUTION)

nx.set_node_attributes(G, final_partition_dict, 'cluster')
filtered_nodes = final_partition_dict.keys()
G = G.subgraph(filtered_nodes)

num_clusters = len(set(final_partition_dict.values()))

labels = list(final_partition_dict.values())
cluster_counts = pd.Series(labels).value_counts()
print(cluster_counts.head(20))

counts_df = cluster_counts.reset_index()
counts_df.columns = ['community_id', 'gene_count']
counts_output_path = os.path.join(OUTPUT_DIR, 'community_gene_counts_HYBRID_HOMOLOGY.csv')
counts_df.to_csv(counts_output_path, index=False)

mapping_df = pd.DataFrame(final_partition_dict.items(), columns=['gene', 'community_id'])
mapping_output_path = os.path.join(OUTPUT_DIR, 'gene_to_community_mapping_HYBRID_HOMOLOGY.csv')
mapping_df.to_csv(mapping_output_path, index=False)
print(f"saved: '{mapping_output_path}'")

communities_dict = defaultdict(list)
for gene, community_id in final_partition_dict.items():
    communities_dict[community_id].append(gene)
communities_dict = dict(communities_dict)

communities_output_path = os.path.join(OUTPUT_DIR,
                                       f'communities_to_genes_{MIN_CLUSTER_SIZE}_{MAX_CLUSTER_SIZE}_HYBRID_HOMOLOGY.json')
try:
    with open(communities_output_path, 'w', encoding='utf-8') as f:
        json.dump(communities_dict, f, ensure_ascii=False, indent=4)
    print(f"communities_output_path: '{communities_output_path}'")
except Exception as e:
    print(f"Error: {e}")

