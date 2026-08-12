import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from index.embedding_datasets import EmbDataset, MultiEmbDataset
from index.models.rqvae import RQVAE


def evaluate_metrics(model, data_loader, device):
    """
    评测模型，计算碰撞率和码本利用率。

    参数:
        model: 要评测的模型。
        data_loader: 数据加载器。
        device: 运行设备。

    返回:
        tuple: (碰撞率, 平均码本利用率, 每层利用率详情)。
    """
    model.eval()
    model = model.to(device)

    all_indices = []
    num_sample = 0

    for data in tqdm(data_loader, desc="Generating indices for evaluation"):
        num_sample += len(data)
        data = data.to(device)
        indices = model.get_indices(data)
        indices = indices.view(-1, indices.shape[-1]).cpu()
        all_indices.append(indices)

    all_indices = torch.cat(all_indices, dim=0).numpy()

    # --- 1. 计算碰撞率 (Collision Rate) ---
    indices_set = set()
    for index in all_indices:
        code = "-".join([str(int(_)) for _ in index])
        indices_set.add(code)

    collision_rate = (num_sample - len(indices_set)) / num_sample

    # --- 2. 计算码本利用率 (Codebook Utilization) ---
    num_quantizers = all_indices.shape[1]
    num_emb_list = model.num_emb_list
    utilizations = []
    detailed_utilization = {}

    for i in range(num_quantizers):
        unique_codes = np.unique(all_indices[:, i])
        utilization = len(unique_codes) / num_emb_list[i]
        utilizations.append(utilization)
        detailed_utilization[f"Layer {i}"] = {
            "used_codes": len(unique_codes),
            "total_codes": num_emb_list[i],
            "utilization": utilization,
        }

    avg_utilization = np.mean(utilizations)

    return collision_rate, avg_utilization, detailed_utilization


def main(args):
    device = torch.device(args.device)

    # 加载模型检查点
    print(f"Loading checkpoint from: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    model_args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    # 从检查点中保存的参数加载原始数据集
    data_paths = getattr(model_args, "data_paths", None)
    if data_paths:
        missing = [p for p in data_paths if not os.path.exists(p)]
        if missing:
            raise ValueError(
                "The model checkpoint contains 'data_paths' but some paths are missing: " + ", ".join(missing)
            )
        print("Loading dataset from (multi):")
        for p in data_paths:
            print(f"  - {p}")
        data = MultiEmbDataset(data_paths)
    else:
        if not hasattr(model_args, "data_path") or not os.path.exists(model_args.data_path):
            raise ValueError(
                "The model checkpoint does not contain a valid 'data_path'. "
                "Cannot run evaluation without the original dataset."
            )
        print(f"Loading dataset from: {model_args.data_path}")
        data = EmbDataset(model_args.data_path)

    # 初始化 RQVAE 模型
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=model_args.num_emb_list,
        e_dim=model_args.e_dim,
        layers=model_args.layers,
        dropout_prob=model_args.dropout_prob,
        bn=model_args.bn,
        loss_type=model_args.loss_type,
        quant_loss_weight=model_args.quant_loss_weight,
        beta=getattr(model_args, "beta", 0.25),  # 兼容旧的模型检查点
        kmeans_init=model_args.kmeans_init,
        kmeans_iters=model_args.kmeans_iters,
        sk_epsilons=model_args.sk_epsilons,
        sk_iters=model_args.sk_iters,
    )

    model.load_state_dict(state_dict)

    # 创建 DataLoader
    data_loader = DataLoader(
        data,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # 执行评测
    collision_rate, avg_utilization, detailed_utilization = evaluate_metrics(model, data_loader, device)

    # 打印结果
    print("\n" + "=" * 60)
    print(" " * 18 + "Evaluation Results")
    print("=" * 60)
    print(f"{'Checkpoint:':<30} {args.ckpt_path}")
    print(f"{'Dataset:':<30} {model_args.data_path}")
    print("-" * 60)
    print(f"{'Collision Rate:':<30} {collision_rate:.6f}")
    print(f"{'Average Codebook Utilization:':<30} {avg_utilization:.6f}")
    print("-" * 60)
    print("Detailed Codebook Utilization per Layer:")
    for layer, stats in detailed_utilization.items():
        print(f"  - {layer}: {stats['utilization']:.4f} ({stats['used_codes']} / {stats['total_codes']} codes used)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RQ-VAE model checkpoint for collision and codebook utilization."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (e.g., 'cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size for evaluation.",
    )
    args = parser.parse_args()
    main(args)
