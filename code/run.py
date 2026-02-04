import os
import sys
import argparse
import gc
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import pickle
import h5py
import torch
from torch import nn
from collections import OrderedDict

from PIL import Image
import cv2
from anndata import AnnData
from scipy.stats import zscore
from pathlib import Path
import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
sys.path.append(current_dir)

from model import load_ddp_checkpoint, save_ddp_checkpoint
from model import config as CFG
from model import (
    STain,STain_test_dataloader,
    get_transforms, CLIPDataset_sc, NumpyDataset, FixedNumpyDataset,
    TextEncoder, AvgMeter, get_lr
)

import scanpy as sc
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import subprocess
import collections
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as pth_transforms
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler

from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import zscore
from statistics import mean

mp.set_start_method('spawn', force=True)

import warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).")

def get_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    if total_params >= 1e9:
        print(f"Model size: {total_params/1e9:.2f}B parameters")
    elif total_params >= 1e6:
        print(f"Model size: {total_params/1e6:.2f}M parameters")
    elif total_params >= 1e3:
        print(f"Model size: {total_params/1e3:.2f}K parameters")
    
    return total_params, trainable_params


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def setup_distributed(backend="gloo", port=29500):  
    """Initialize distributed training environment."""
    
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    
    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        os.environ["MASTER_ADDR"] = addr
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    
    os.environ["MASTER_PORT"] = str(port)
    
    torch.cuda.set_device(rank)
    
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=120)
    )
    
    print(f"Initialized process group with {backend}: rank {rank}, world_size {world_size}")
    return rank

def build_loaders(train_dir, test_dir, batch_size, num_workers,debug,mode,tissue='INT6'):
    print(f'build loaders {tissue}')
    transforms = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),  
        pth_transforms.RandomHorizontalFlip(p=0.5), 
        pth_transforms.RandomVerticalFlip(p=0.5),   
        pth_transforms.ColorJitter(                 
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02
        ),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
    ])

    h5_train_valid = h5py.File(train_dir,'r')
    h5_test = h5py.File(test_dir,'r')
    train_len = h5_train_valid['barcode'].shape[0]
    test_len = h5_test['barcode'].shape[0]
    print(f'debug: {debug}')
    if mode == 'test':
        if debug is False:
            max_id = train_len
            test_maxid = test_len
        else:
            max_id = 100
            test_maxid = test_len
        test_ids = np.arange(0, test_maxid)
        sample_ids = []
        for i in test_ids:
            if h5_test['barcode'][i].decode('utf-8').startswith(tissue):
                sample_ids.append(i)
        test_ids = sample_ids
        h5_train_valid.close()
        h5_test.close()
        testset = FixedNumpyDataset( #dataloadertest_ids
            [idx for idx in test_ids],
            test_dir,
            'test')
        test_sampler = torch.utils.data.distributed.DistributedSampler(
                    testset, 
                    shuffle=False
                )

        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=test_sampler,  
            pin_memory=True,       
            drop_last=False       
        )
        return testloader


def plotdecoder(decoder_losses, epoch, experiment):
    plt.plot(decoder_losses)
    plt.title('Decoder Loss over time')
    plt.xlabel('Step')
    plt.ylabel('Decoder Loss')
    plt.savefig(f'decoder_loss_plot_{epoch}_{experiment}.png')

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='gelu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

def test_epoch(model, epoch, test_loader, test_dir,step_2=2):
    try:
        from skimage.metrics import structural_similarity as ssim_func
    except ImportError:
        ssim_func = None
    model.eval()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    tqdm_object = tqdm(test_loader, total=len(test_loader), desc=f"Testing Epoch {epoch}") if rank == 0 else test_loader
    h5_file = h5py.File(test_dir, 'r')
    
    local_decoded = []
    local_target = []
    local_indices = [] 
    local_barcodes = []
    local_coords = []

    with torch.no_grad():
        for batch in tqdm_object:
            batch = [b.cuda(non_blocking=True) if torch.is_tensor(b) else b for b in batch]
            
            model_input = batch[:-1]
            target_y = batch[2] 
            indices = batch[-1]
            indices_np = indices.cpu().numpy()

            batch_barcodes = []
            for idx in indices_np:
                barcode = h5_file['barcode'][idx]
                if isinstance(barcode, bytes):
                    barcode = barcode.decode('utf-8')
                batch_barcodes.append(barcode)
            batch_coords = []
            for idx in indices_np:
                if 'coords' in h5_file:
                    coord = h5_file['coords'][idx]
                else:
                    coord = np.array([0, 0])  
                batch_coords.append(coord)
            recon_y = model(model_input, epoch, step_2=step_2, use_gene_expression=False, weighted_loss=False)
            
            local_decoded.append(recon_y)
            local_target.append(target_y)
            local_indices.append(indices)
            local_barcodes.extend(batch_barcodes)
            local_coords.extend(batch_coords)
        h5_file.close()
            
    local_decoded = torch.cat(local_decoded, dim=0)
    local_target = torch.cat(local_target, dim=0)
    local_indices = torch.cat(local_indices, dim=0)
    local_coords_array = np.array(local_coords)

    gathered_decoded = [torch.zeros_like(local_decoded) for _ in range(world_size)]
    gathered_target = [torch.zeros_like(local_target) for _ in range(world_size)]
    gathered_indices = [torch.zeros_like(local_indices) for _ in range(world_size)]

    
    dist.all_gather(gathered_decoded, local_decoded)
    dist.all_gather(gathered_target, local_target)
    dist.all_gather(gathered_indices, local_indices)

    gathered_barcodes = [None for _ in range(world_size)]
    gathered_coords = [None for _ in range(world_size)]

    dist.all_gather_object(gathered_barcodes, local_barcodes)
    dist.all_gather_object(gathered_coords, local_coords_array.tolist())  
    if rank == 0:
        full_decoded = torch.cat(gathered_decoded, dim=0)
        full_target = torch.cat(gathered_target, dim=0)
        full_indices = torch.cat(gathered_indices, dim=0)

        all_barcodes = []
        for b_list in gathered_barcodes:
            if b_list is not None:
                all_barcodes.extend(b_list)

        all_coords = []
        for c_list in gathered_coords:
            if c_list is not None:
                all_coords.extend(c_list)
        all_coords = np.array(all_coords)


        _, sort_idx = torch.sort(full_indices)
        full_decoded = full_decoded[sort_idx]
        full_target = full_target[sort_idx]

        sorted_barcodes = [all_barcodes[i] for i in sort_idx.cpu().numpy() if i < len(all_barcodes)]
        sorted_coords = all_coords[sort_idx.cpu().numpy()[:len(sorted_barcodes)]]
    
        actual_size = len(test_loader.dataset)
        full_decoded = full_decoded[:actual_size].cpu().numpy()
        full_target = full_target[:actual_size].cpu().numpy()
        sorted_barcodes = sorted_barcodes[:actual_size]
        sorted_coords = sorted_coords[:actual_size]

        print(f"Calculating metrics for {actual_size} samples...")
        num_genes = full_decoded.shape[1]
        
        pcc = []
        srcc = []
        rmse = []
        ssims = []

        for i in range(num_genes):
            d = full_decoded[:, i]
            t = full_target[:, i]
            
            # Pearson
            p_val, _ = pearsonr(d, t)
            pcc.append(p_val)
            
            # Spearman
            s_val, _ = spearmanr(d, t)
            srcc.append(s_val)
            
            # RMSE
            rmse.append(sqrt(mean_squared_error(d, t)))
            
            if ssim_func:
                ssims.append(ssim_func(d, t, data_range=7))
            else:
                ssims.append(0)

        return ([pcc, srcc, rmse, ssims], full_decoded, full_target, sorted_barcodes, sorted_coords)
    
    return None, None, None, None, None

def default_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

def parse_args():
    parser = argparse.ArgumentParser(description='Training script with configurable parameters')
    parser.add_argument('--model_name', type=str, 
                       default='STain_hest1k',
                       help='moedel name')
    
    parser.add_argument('--mode', type=str, choices=['train','test'], required=True, default='train', help='train or test')
    parser.add_argument('--debug', type=str, default='False', choices=['True', 'False'], help='debug or not')
    parser.add_argument('--train_dir', type=str, 
                       default='/jizhi/jizhi2/worker/trainer/youngeegu/datasets/hest1k/hest_data/hest1k_whole/e25/hest1k_cnts_train_e25.h5',
                       help='Path to training data H5 file')
    parser.add_argument('--test_dir', type=str,
                       default='/jizhi/jizhi2/worker/trainer/youngeegu/datasets/hest1k/hest_data/hest1k_whole/e25/hest1k_cnts_test_e25.h5',
                       help='Path to test data H5 file')
    parser.add_argument('--model_path', type=str,
                       default='experiments/hest1k/stain_nbrs_diam/best_model.pt',
                       help='Path to pretrained model')
    parser.add_argument('--save_dir', type=str,
                       default='/jizhi/jizhi2/worker/trainer/youngeegu/projects/STain/results/STain_hest1k',
                       help='Path to save results')
    parser.add_argument('--gene_list_path', type=str,
                       default='/jizhi/jizhi2/worker/trainer/youngeegu/datasets/hest1k/hest_data/hvg_5033.npy',
                       help='Path to gene list file')
    
    parser.add_argument('--epochs', type=int, default=CFG.epochs, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=CFG.batch_size, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=CFG.num_workers, help='Number of data loader workers')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay')
    parser.add_argument('--step_2', type=int, default=2, choices=[1, 2], help='Training step phase')
    parser.add_argument('--patience', type=int, default=CFG.patience, help='Patience for learning rate scheduler')
    parser.add_argument('--factor', type=float, default=CFG.factor, help='Factor for learning rate scheduler')
    
    parser.add_argument('--CLIP', type=str, default='False', choices=['True', 'False'], help='Use CLIP loss')
    parser.add_argument('--nbrs', type=str, default='False', choices=['True', 'False'], help='Use neighbor features')
    parser.add_argument('--FiLM', type=str, default='False', choices=['True', 'False'], help='Use FiLM conditioning')
    parser.add_argument('--device', type=str, default=CFG.device, help='Device to use (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16'], help='Data type')
    parser.add_argument('--weighted_loss', action='store_true', default=CFG.weighted_loss, help='Use weighted loss')

    #parser.add_argument('--experiment', type=str, default='hest1k', help='Experiment name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    
    # Set random seed
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.dtype == 'float16':
        dtype = torch.float16
    else:
        dtype = torch.float32

    print("Initializing model...")

    if args.mode == 'test':
        use_gene_expression = False

        model = STain(CLIP=args.CLIP, nbrs=args.nbrs, FiLM=args.FiLM).to(dtype)

        if args.model_path and os.path.exists(args.model_path):
            print(f"Loading pretrained model from: {args.model_path}")

            device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

            start_epoch, loss, info = load_ddp_checkpoint(
                        model=model,
                        checkpoint_path=args.model_path,
                        device=f'cuda:{local_rank}',
                        strict=True,
                        optimizer=None,    
                        lr_scheduler=None 
                    )
        model = model.to(CFG.device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank, 
            find_unused_parameters=True
        )
        model.eval()
    
        print(f"✅ Model ready, using device: {CFG.device}")

        test_files2=['TENX92' ]

        if dist.get_rank() == 0:
            pcc_dict = {}

        for i in test_files2:
            epoch = 'best_model'

            if dist.get_rank() == 0:
                print(f'\n{"="*60}')
                print(f'We are now at {i}')
                print(f'{"="*60}')

            tissue = i.replace('.h5ad','')

            test_loader = build_loaders(
                args.train_dir, args.test_dir, args.batch_size, args.num_workers,
                debug=False, mode=args.mode, tissue=tissue
            )

            if len(test_loader.dataset) == 0:
                if dist.get_rank() == 0:
                    print(f"No samples found for tissue: {tissue}")
                continue

            (evaldf, expresstiondf, gtlist,barcodelist,coordslist) = test_epoch(model, epoch, test_loader,args.test_dir)
            if dist.get_rank() == 0:
                if evaldf and len(evaldf[0]) > 0 and expresstiondf is not None and gtlist is not None:
                    transposed_list = list(map(list, zip(*evaldf)))
                    df = pd.DataFrame(transposed_list, columns=['pcc','srcc','rmse','ssim'])
                    gene_list = list(np.load('./example/hvg_5033.npy', allow_pickle=True))

                    if len(gene_list) >= len(df):
                        df['gene'] = gene_list[:len(df)]
                    else:
                        print(f"Warning: Gene list length ({len(gene_list)}) doesn't match results ({len(df)})")
                        df['gene'] = gene_list + ['unknown'] * (len(df) - len(gene_list))

                    save_root = Path(args.save_dir)
                    save_root.mkdir(parents=True, exist_ok=True)

                    csv_path = save_root / f'{tissue}_{epoch}.csv'
                    df.to_csv(csv_path, index=False)
                    print(f'Saved evaluation to: {csv_path}')

                    expressionndf = np.array(expresstiondf)
                    gtdf = np.array(gtlist)
                    barcodelist = np.array(barcodelist)
                    coordslist = np.array(coordslist)

                    print(f'Prediction shape: {expressionndf.shape}')
                    print(f'Ground truth shape: {gtdf.shape}')

                    np.save(save_root / f'{tissue}_{epoch}_prediction.npy', expressionndf)
                    np.save(save_root / f'{tissue}_{epoch}_groundtruth.npy', gtdf)
                    np.save(save_root / f'{tissue}_{epoch}_coords.npy', coordslist)

                    print(f'Prediction mean: {np.mean(expressionndf):.4f}')
                    print(f'Prediction min: {np.min(expressionndf):.4f}')
                    print(f'Prediction max: {np.max(expressionndf):.4f}')

                    cur_pcc = [x for x in df['pcc'] if not np.isnan(x)]
                    avg_pcc = sum(cur_pcc) / len(cur_pcc) if cur_pcc else 0
                    pcc_dict[tissue] = avg_pcc

                    print(f'Average PCC for {tissue}: {avg_pcc:.4f}')
                    print(f'Number of genes evaluated: {len(cur_pcc)}')
                else:
                    print(f"No valid results for tissue: {tissue}")

            dist.barrier()

        if dist.get_rank() == 0 and pcc_dict:
            print(f'\n{"="*60}')
            print('SUMMARY OF ALL TISSUES:')
            print(f'{"="*60}')

            summary_df = pd.DataFrame({
                'tissue': list(pcc_dict.keys()),
                'avg_pcc': list(pcc_dict.values())
            })

            summary_df = summary_df.sort_values('avg_pcc', ascending=False)

            print("\nTissue Performance (sorted by PCC):")
            for idx, row in summary_df.iterrows():
                print(f"  {row['tissue']}: {row['avg_pcc']:.4f}")
            overall_avg = summary_df['avg_pcc'].mean()
            print(f'\nOverall average PCC across {len(summary_df)} tissues: {overall_avg:.4f}')

            summary_path = Path(args.save_dir) / 'summary_results_.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f'\nSummary saved to: {summary_path}')

if __name__ == "__main__":
    local_rank = setup_distributed()
    #local_rank, device = setup_distributed(backend="nccl", port=29501)
    main()