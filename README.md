# scGenoByte: A GenoByte Embedding Transformer with Biological Priors for Cell Type Annotation



**scGenoByte** is a unified deep learning framework designed to enhance single-cell representation learning through biologically informed full-gene modeling. 

------

## 🌟 Key Innovations

- **Biologically Coherent Tokenization Termed Genobytes**
- **Genome-wise Modeling Framework**
- **Proteomic-Driven Semantic Alignment**
- **Pathway-Guided Regularization** 

------

## 📂 Repository Structure

```
scGenoByte/
├── visualization/
	├── genobyte_eval_enrich.py
	├── genobyte_eval_ppi.py
	└── genobyte_protein_alignment.py  
├── util/
├── GenoByte_construction.py   
├── models_pretrain.py        
├── models_finetune.py         
├── engine_pretrain.py         
├── engine_finetune.py
├── main_finetune.py
├── main_pretrain.py
├── preprocess.py
├── sc_patchemb.py
├── mydataset.py
├── enrich_statistical.py
└── requirements.txt           
```

------

## 📊 Data Preparation

The required datasets and biological priors for **scGenoByte** are hosted on Quark Drive. You can download the complete data package from the link below:

> 🔗 **Data Download Link:** [https://pan.quark.cn/s/fe21d768b090](https://pan.quark.cn/s/f29f79d91b2a)

### 📂 Directory Structure

Plaintext

```
scGenoByte_Data/
├── down-stream datasets/       # Benchmarking datasets for cell type annotation
│   ├── Baron/                  
│   ├── Lung/                   
│   ├── MacParland/             
│   ├── Muraro/                 
│   ├── Pan-GI/                 
│   ├── Segerstolpe/            
│   ├── Xin/                    
│   └── Zheng68K/               
├── esm_embedding/              # Protein squences embedding (via ESM-2/ESM-1b)
│   ├── Homo_sapiens.GRCh38.pep.all.clean.fa
│   ├── Homo_sapiens.GRCh38.pep.all.fa
│   ├── Homo_sapiens.GRCh38.pep.all.gene_symbol_to_embedding_ESM1b.pt
│   └── Homo_sapiens.GRCh38.pep.all.gene_symbol_to_protein_ID.json
├── HOMOLOGY/                   # Gene paralogy information
│   └── HOMOLOGY_FILE.txt       
├── model/                      # Model checkpoints
│   └── pretrain_model/         
└── ppi_network/                # Biological interaction priors
    └── format_h_sapiens.csv    
```

------

### 📝 Dataset Descriptions

| **Category**                    | **Component**          | **Description**                                              |
| ------------------------------- | ---------------------- | ------------------------------------------------------------ |
| **Benchmark**                   | `down-stream datasets` | Contains 8 standardized datasets used for cross-validation.  |
| **Protein Sequences Embedding** | `esm_embedding`        | Pre-computed embeddings from ESM models to support topological semantic alignment. |
| **Homology**                    | `HOMOLOGY`             | Parsed homology data used to calculate the edge weights for GenoByte construction. |
| **PPI**                         | `ppi_network`          | protein-protein interaction scores.                          |
| **Weights**                     | `model`                | Pre-trained scGenoByte parameters.                           |

------

## 🚀 Quick Start

### 1. Installation

```
git clone git@github.com:yjs193/scGenoByte.git
cd scGenoByte
pip install -r requirements.txt
```

### 2. GenoByte Construction

```
python GenoByte_construction.py
```

### 3. Training & Annotation

**Pre-training:**

```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 main_pretrain.py \
--data_path your_data_path.h5ad \
--protein_embed_path your_esm_embedding_path.pt \
--batch_size 128 \
--epoch 200 \
--output_dir ./output
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main_pretrain.py ... # multi gpus
```

**Fine-tuning (e.g., Pan-GI dataset):**

```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 main_finetune.py \  
--data_path your_data_path.h5ad \ 
--finetune your_model_checkpoint_path.h5ad \
--batch_size 128 \
--epoch 100 \
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main_finetune.py ... # multi gpus
```

## 📧 Contact

For questions, please contact J. Yao (csyjs@mail.scut.edu.cn)
