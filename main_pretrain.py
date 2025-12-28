import argparse
import datetime
import wandb
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import scanpy as sc
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from utils import set_seed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.utils.data import DataLoader
import models_pretrain
from engine_pretrain import train_one_epoch
from mydataset import SCDataset_all
from scipy.sparse import hstack
import scipy.sparse as sp
import platform


def supplydata(data):
    patch_num = 16
    i = data.X.shape[1] % patch_num
    if i == 0:
        return data.X
    supply = patch_num - i
    supplypart = np.zeros([data.X.shape[0], supply], dtype=np.float32)
    supplypart_sparse = sp.csr_matrix(supplypart)

    return hstack([data.X, supplypart_sparse])


def get_args_parser():
    parser = argparse.ArgumentParser('scGenoByte pre-training', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, )
    parser.add_argument('--model', default='mae_vit_patch16_dim256', type=str, metavar='MODEL', )
    parser.add_argument('--input_size', default=25648, type=int, )
    parser.add_argument('--mask_ratio', default=0.625, type=float, )
    parser.add_argument('--clip_grad', default=2.0, type=float, )
    parser.add_argument('--norm_pix_loss', action='store_true', )
    parser.set_defaults(norm_pix_loss=False)
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=3e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    # save and resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=15, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--distributed', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--protein_embed_path', type=str,
                        default='/data/js/data/esm_embedding/Homo_sapiens.GRCh38.pep.all.gene_symbol_to_embedding_ESM1b.pt',)
    parser.add_argument('--data_path', type=str,
                        default='/data/js/data/paper_data/panglao/panglao_merge_23856.h5ad',)
    parser.add_argument('--similarity_loss_weight', type=float, default=0.1,
                        help='Weight for the protein similarity loss')
    parser.add_argument('--standardize_targets', action='store_true',
                        help='Apply Z-score standardization to the target protein patch embeddings')
    parser.set_defaults(standardize_targets=True)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--mixup_alpha', default=0, type=float, help='mixup interpolation coefficient (default: 1.0)')
    args = parser.parse_args()
    return args, parser


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    args.device = 'cuda'
    device = torch.device(args.device)
    set_seed(args.seed)

    cudnn.benchmark = True
    train_adata = sc.read(args.data_path)

    args.standardize_targets = True

    print(f"Loading protein embeddings from {args.protein_embed_path}...")
    protein_embed_dict = torch.load(args.protein_embed_path, map_location='cpu')

    patch_size = 16
    print(f"Using patch size: {patch_size}")
    protein_embed_dim = list(protein_embed_dict.values())[0].shape[0]
    print(f"Loaded protein embeddings for {len(protein_embed_dict)} genes. Embedding dim: {protein_embed_dim}")

    gene_names = train_adata.var_names.to_numpy()
    num_genes = train_adata.shape[1]
    num_patches = num_genes // patch_size

    assert num_genes % patch_size == 0, f"Number of genes ({num_genes}) is not divisible by patch size ({patch_size})"

    gene_to_embedding = torch.zeros((num_genes, protein_embed_dim), dtype=torch.float32)
    for i, gene in enumerate(gene_names):
        if gene in protein_embed_dict:
            gene_to_embedding[i] = protein_embed_dict[gene]

    gene_to_embedding_patched = gene_to_embedding.view(num_patches, patch_size, protein_embed_dim)
    patch_protein_targets = torch.max(gene_to_embedding_patched, dim=1).values

    if args.standardize_targets:
        print("Applying Z-score standardization to patch protein targets...")
        mean = patch_protein_targets.mean(dim=0, keepdim=True)
        std = patch_protein_targets.std(dim=0, keepdim=True)
        patch_protein_targets = (patch_protein_targets - mean) / (std + 1e-6)
    else:
        print("Using raw (non-standardized) patch protein targets.")

    train_data = train_adata.X
    train_dataset = SCDataset_all(train_data, patch_protein_targets)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    train_sampler = torch.utils.data.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)

    args.input_size = train_data.shape[1]

    model = models_pretrain.__dict__[args.model](
        gene_size=args.input_size,
        norm_pix_loss=args.norm_pix_loss,
        protein_embed_dim=protein_embed_dim,
        similarity_loss_weight=args.similarity_loss_weight
    ).to(device)

    print("Model = %s" % str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    if global_rank == 0:
        wandb_logger = wandb.init(project='scGenoByte', config=args.__dict__, name=day, group='pretrain')
    else:
        wandb_logger = None
    if args.clip_grad is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_one_epoch(
            model, train_loader,
            optimizer, device, epoch, loss_scaler,
            wandb_logger=wandb_logger,
            args=args, max_norm=args.clip_grad
        )
        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch)

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('Training time {}'.format(total_time_str))
    wandb.finish()


if __name__ == '__main__':
    current_os = platform.system()

    if current_os == "Windows":
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    torch.multiprocessing.set_start_method('spawn')
    args, parser = get_args_parser()

    day = f"{datetime.datetime.now().strftime('%m-%d')}/{args.model}_{args.model.split('patch')[-1]}"
    args.output_dir += day
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)