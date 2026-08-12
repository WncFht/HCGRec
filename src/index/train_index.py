import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from index.embedding_datasets import EmbDataset, MultiEmbDataset
from index.models.rqvae import RQVAE
from index.trainer import Trainer


def _validate_args(args):
    if len(args.num_emb_list) != len(args.sk_epsilons):
        raise ValueError(
            "Length mismatch: --num_emb_list and --sk_epsilons must have "
            f"the same length, got {len(args.num_emb_list)} vs {len(args.sk_epsilons)}"
        )


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        v = str(v).strip().lower()
        if v in {"1", "true", "t", "yes", "y"}:
            return True
        if v in {"0", "false", "f", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
    )
    parser.add_argument("--eval_step", type=int, default=50, help="eval step")
    parser.add_argument("--learner", type=str, default="AdamW", help="optimizer")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="scheduler")
    parser.add_argument("--warmup_epochs", type=int, default=50, help="warmup epochs")
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data_path",
        type=str,
        default="../data/Games/Games.emb-llama-td.npy",
        help="Input data path.",
    )
    data_group.add_argument(
        "--data_paths",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of input .npy paths; if provided, train on the union of all datasets.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="l2 regularization weight",
    )
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=str2bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument(
        "--kmeans_init",
        type=str,
        default="True",
        help="Use kmeans_init or not ('True' or 'False').",
    )
    parser.add_argument(
        "--large_scale_kmeans",
        type=str,
        default="True",
        help="Use large scale data for kmeans_init ('True' or 'False').",
    )
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument(
        "--sk_epsilons",
        type=float,
        nargs="+",
        default=[0.0, 0.0, 0.0],
        help="sinkhorn epsilons",
    )
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")

    parser.add_argument(
        "--num_emb_list",
        type=int,
        nargs="+",
        default=[256, 256, 256],
        help="emb num of every vq",
    )
    parser.add_argument("--e_dim", type=int, default=32, help="vq codebook embedding size")
    parser.add_argument(
        "--quant_loss_weight",
        type=float,
        default=1.0,
        help="vq quantion loss weight",
    )
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[2048, 1024, 512, 256, 128, 64],
        help="hidden sizes of every layer",
    )

    parser.add_argument("--save_limit", type=int, default=5)
    parser.add_argument("--ckpt_dir", type=str, default="", help="output directory for model")
    parser.add_argument("--use_wandb", type=str2bool, default=False, help="use wandb or not")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="grec_index",
        help="wandb project",
    )
    parser.add_argument("--wandb_name", type=str, default=None, help="wandb run name")
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional experiment/run name to persist into run_meta.json.",
    )

    return parser.parse_args()


def _init_distributed(args):
    if not dist.is_available():
        return {"enabled": False, "rank": 0, "world_size": 1, "local_rank": 0}

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return {"enabled": False, "rank": 0, "world_size": 1, "local_rank": 0}

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    else:
        args.device = "cpu"

    return {
        "enabled": True,
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "local_rank": local_rank,
    }


def clean_dataset_embeddings(dataset):
    """
    清理数据集中的 NaN / Inf 样本（如果数据集有 .embeddings 属性）。
    会就地修改 dataset.embeddings（以及可选的 dataset.ids），返回 (dataset, 删除行数)。
    """
    if not hasattr(dataset, "embeddings"):
        return dataset, 0

    emb = dataset.embeddings
    num_bad = 0
    mask = None

    if isinstance(emb, np.ndarray):
        mask = np.isfinite(emb).all(axis=1)
        num_bad = int(emb.shape[0] - mask.sum())
        if num_bad > 0:
            print(f"[warn] Dropping {num_bad} samples with NaN/Inf from dataset.embeddings (numpy).")
            dataset.embeddings = emb[mask]
    elif torch.is_tensor(emb):
        flat_finite = torch.isfinite(emb).view(emb.size(0), -1)
        mask = flat_finite.all(dim=1)
        num_bad = int(emb.size(0) - int(mask.sum().item()))
        if num_bad > 0:
            print(f"[warn] Dropping {num_bad} samples with NaN/Inf from dataset.embeddings (tensor).")
            dataset.embeddings = emb[mask]
    else:
        # 未知类型就直接跳过
        return dataset, 0

    # 同步 ids（如果存在且长度对得上）
    if mask is not None and hasattr(dataset, "ids"):
        ids = dataset.ids
        if isinstance(ids, (list, np.ndarray)) and len(ids) == len(mask):
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = mask
            dataset.ids = [id_ for id_, keep in zip(ids, mask_np, strict=False) if keep]

    return dataset, num_bad


if __name__ == "__main__":
    """fix the random seed"""
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    ddp = _init_distributed(args)
    is_main_process = (not ddp["enabled"]) or ddp["rank"] == 0
    # 将字符串参数转换为布尔值
    args.kmeans_init = args.kmeans_init.lower() == "true"
    args.large_scale_kmeans = args.large_scale_kmeans.lower() == "true"
    _validate_args(args)

    if is_main_process:
        print("=================================================")
        print(args)
        print("=================================================")

    logging.basicConfig(level=logging.DEBUG)

    """build dataset"""
    train_paths = args.data_paths if args.data_paths is not None else [args.data_path]
    data = MultiEmbDataset(train_paths) if len(train_paths) > 1 else EmbDataset(train_paths[0])

    # 训练前清理数据中的 NaN / Inf 样本（如果数据集支持）
    data, num_bad_global = clean_dataset_embeddings(data)
    if is_main_process and num_bad_global > 0:
        print(f"[info] Dropped {num_bad_global} invalid samples from training dataset.")

    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )

    if ddp["enabled"]:
        device = torch.device(args.device)
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[ddp["local_rank"]] if device.type == "cuda" else None,
        )

    if is_main_process:
        print(model)

    sampler = DistributedSampler(data, shuffle=True) if ddp["enabled"] else None
    data_loader = DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        pin_memory=True,
    )
    trainer = Trainer(args, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)

    if is_main_process:
        print("Best Loss", best_loss)
        print("Best Collision Rate", best_collision_rate)

    if ddp["enabled"]:
        dist.destroy_process_group()
