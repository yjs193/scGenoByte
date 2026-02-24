import argparse
import platform
import numpy as np
import os
import time
from pathlib import Path
import scanpy as sc
import copy
import datetime
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup  #
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import scipy.sparse as sp
from scipy.sparse import hstack
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from tqdm import tqdm
import models_finetune
import util.lr_decay as lrd
import util.misc as misc
from preprocess import split_h5ad
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from utils import set_seed
from mydataset import SCDataset
import pandas as pd
from engine_finetune import train_one_epoch, evaluate
import wandb


def supplydata(data):
    i = data.X.shape[1] % args.patch_size
    if i == 0:
        return data.X
    supply = args.patch_size - i
    supplypart = np.zeros([data.X.shape[0], supply], dtype=np.float32)
    supplypart_sparse = sp.csr_matrix(supplypart)

    return hstack([data.X, supplypart_sparse])



def get_args_parser():
    parser = argparse.ArgumentParser('scGenoByte fine-tuning for cell type annotation', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--patch_size', default=16, type=int, help='patch size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--clip_grad', type=float, default=2.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=3e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr dedcay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=6e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--pathway_loss_beta', type=float, default=2.0,
                        help='Weight for the pathway reconstruction loss (beta)')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--finetune',
                        default=r'/data/js/data/model/PII_cluster/11-12/mae_vit_patch16_dim256_ep200_16_dim256/checkpoint-199.pth')
    parser.add_argument('--dataset_name', default='empty', type=str, help='dataset name')
    parser.add_argument('--train_base', action='store_true')
    parser.add_argument('--mixup', type=float, default=0, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='Number of neighbors for UMAP construction')
    parser.add_argument('--cutmix', type=float, default=0, help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None, )
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--data_path', default='/data/js/data/paper_data/raw/Zheng68k/Zheng68K_23856_with_pathways_Reactome_v3_stat.h5ad', type=str, help='dataset path')
    parser.add_argument('--nb_classes', default=11, type=int, help='number of the classification types')
    parser.add_argument('--pretrain_epoch', default=0, type=int, help='pretrain_epoch')
    parser.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--save_ckpt_freq', default=1, type=int, metavar='N', help='save epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default='', help='url used to set up distributed training')

    # 【新增参数】
    parser.add_argument('--cluster_test', action='store_true',
                        help='Enable clustering evaluation (ARI/NMI + UMAP) during training')
    args = parser.parse_args()
    return parser, args


def main(args):
    misc.init_distributed_mode(args)
    if platform.system() == "Windows":
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12340'
        dist.init_process_group(backend='GLOO', init_method='env://', rank=0, world_size=1)

    device = torch.device(args.device)
    cudnn.benchmark = True
    print("{}".format(args).replace(', ', ',\n'))
    set_seed(args.seed)

    global_rank = misc.get_rank()
    dataset_path = args.data_path
    args.dataset_name = dataset_path.split('/')[-1].split('_')[0]

    cls_key = 'celltype'
    adata = sc.read_h5ad(dataset_path)

    adata.uns['pathways_mtx_cols'] = [
        str(i)
        if len(str(i)) < 25
        else str(i)[:12] + '...' + str(i)[-12:]
        for i in adata.uns['pathways_mtx_cols']
    ]
    print(adata.shape)

    args.patch_size = int(args.model.split('_')[-1][5:])
    wandb_run_name = f'{args.model}_{args.dataset_name}_{cls_key}'
    add_str = f'{datetime.datetime.now().strftime("%m-%d")}/{wandb_run_name}'
    args.output_dir += add_str
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if 'pathways_mtx' not in adata.obsm:
        raise ValueError("Pathway matrix 'pathways_mtx' not found in adata.obsm. "
                         "Please compute it and add it to the AnnData object before running.")
    adata.obs['celltype'] = adata.obs[cls_key]
    args.num_pathways = adata.obsm['pathways_mtx'].shape[1]
    print(f"Found pathway matrix with {args.num_pathways} pathways.")
    test_size = 0.2
    train_adata, val_adata = split_h5ad(cls_key, adata, test_size=test_size, random_state=2025)
    label_dict_train, label_train = np.unique(np.array(train_adata.obs[cls_key]), return_inverse=True)
    label_dict_val, label_val = np.unique(np.array(val_adata.obs[cls_key]),
                                          return_inverse=True)  # <-- label_dict_val 已定义
    label_train = torch.from_numpy(label_train)
    label_val = torch.from_numpy(label_val)

    train_data = train_adata.X
    val_data = val_adata.X
    pathway_train = torch.from_numpy(train_adata.obsm['pathways_mtx'].astype(np.float32))
    pathway_val = torch.from_numpy(val_adata.obsm['pathways_mtx'].astype(np.float32))

    print(train_data.shape)
    print(pathway_train.shape)
    print(val_data.shape)
    print(pathway_val.shape)

    dataset_train = SCDataset(train_data, label_train, pathway_train)
    dataset_val = SCDataset(val_data, label_val, pathway_val)

    if args.distributed:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    print(train_data.shape[1])
    args.nb_classes = len(label_dict_train)

    model = models_finetune.__dict__[args.model](
        img_size=train_data.shape[1],
        num_classes=args.nb_classes,
        num_pathways=args.num_pathways,  # <-- 传入新参数
        drop_path_rate=args.drop_path,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        trunc_normal_(model.head.weight, std=2e-5)
    elif args.finetune and args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)

    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        criterion_class = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion_class = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion_class = torch.nn.CrossEntropyLoss()

    criterion_recon = torch.nn.SmoothL1Loss()

    print("criterion_class = %s" % str(criterion_class))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        evaluate(data_loader_val, model, criterion_class, criterion_recon, device, args)
        print('Done!')
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_acc, max_f1 = 0.0, 0.0
    if global_rank == 0:
        # wandb_logger = wandb.init(project='scGenoByte', config=args.__dict__, name=wandb_run_name,
        #                           group=f'{args.dataset_name}')  # , settings=wandb.Settings(init_timeout=600)
        wandb_logger = None
    else:
        wandb_logger = None
    best_model_state = None
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model, criterion_class, criterion_recon, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            wandb_logger=wandb_logger,
            args=args
        )

        if args.output_dir and misc.is_main_process() and epoch % args.save_ckpt_freq == 0:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats, f1, acc, truths, preds = evaluate(data_loader_val, model, criterion_class, criterion_recon, device, args)

        is_max_acc_changed = acc > max_acc
        if misc.is_main_process() and (epoch >= 2 and is_max_acc_changed):
            plot_confusion_matrix(data_loader_val, model, device, label_dict_val, args, epoch)
        if acc > max_acc:
            max_acc = acc
            max_f1 = f1
            if misc.is_main_process():
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"New best model found at epoch {epoch} with Acc: {acc:.4f}")

                if args.output_dir:
                    truth_names = [label_dict_val[i] for i in truths]
                    pred_names = [label_dict_val[i] for i in preds]
                    df_results = pd.DataFrame({
                        'truth': truth_names,
                        'prediction': pred_names
                    })
                    save_name = f'best_predictions_epoch_{epoch}_acc_{acc:.4f}.csv'
                    save_path = os.path.join(args.output_dir, save_name)
                    try:
                        df_results.to_csv(save_path, index=False)
                        print(f"Saved best model predictions to: {save_path}")
                    except Exception as e:
                        print(f"Failed to save prediction CSV: {e}")

        if best_model_state is None:
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"New best model found at epoch {epoch} with Acc: {acc:.4f}")
        max_acc = max(max_acc, acc)
        max_f1 = max(max_f1, f1)

        if global_rank == 0 and wandb_logger is not None:
            prefix = f"{args.dataset_name}/"
            dict = {
                prefix + "val_loss": test_stats['loss'],
                prefix + "val_loss_class": test_stats['loss_class'],  # <-- 新增
                prefix + "val_loss_recon": test_stats['loss_recon'],  # <-- 新增
                prefix + "val_acc": acc,
                prefix + "val_f1score": f1,
                prefix + "max_acc": max_acc,
                prefix + "max_f1": max_f1,
            }
            wandb_logger.log(dict)

        print(f"Accuracy of the network on the {len(dataset_val)} test images: {acc:.3f}%")
        print(f'Max accuracy: {max_acc:.2f}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    wandb.finish()


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '8'
    if platform.system() == "Windows":
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    parser, args = get_args_parser()

    main(args)