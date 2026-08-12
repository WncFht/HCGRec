import argparse
import collections
import json
import os
import re
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from index.embedding_datasets import EmbDataset, MultiEmbDataset
from index.models.rqvae import RQVAE


def check_collision(all_keys):
    """
    检查所有索引是否存在碰撞（即重复）。

    参数:
        all_keys: 包含所有索引 key 的数组/列表（例如 tuple(int,...)）。

    返回:
        bool: 如果没有碰撞返回 True，否则返回 False。
    """
    tot_item = len(all_keys)  # 总项目数
    tot_indice = len(set(all_keys))  # 唯一索引的数量
    return tot_item == tot_indice  # 如果总项目数等于唯一索引数，则没有碰撞


def get_indices_count(all_keys):
    """
    计算每个索引出现的次数。

    参数:
        all_keys: 包含所有索引 key 的数组/列表（例如 tuple(int,...)）。

    返回:
        collections.defaultdict: 字典，键为索引字符串，值为其出现次数。
    """
    indices_count = collections.defaultdict(int)  # 使用 defaultdict 方便计数
    for key in all_keys:
        indices_count[key] += 1
    return indices_count


def get_collision_item(all_keys):
    """
    获取所有发生碰撞的项目的分组。

    参数:
        all_keys: 包含所有索引 key 的数组/列表（例如 tuple(int,...)）。

    返回:
        list: 列表中的每个元素是一个列表，包含发生碰撞的项目索引。
    """
    index2id = {}  # 字典，用于存储索引到项目ID的映射
    for i, key in enumerate(all_keys):
        if key not in index2id:
            index2id[key] = []
        index2id[key].append(i)  # 将项目ID添加到对应索引的列表中

    # 只保留有冲突的item（即出现次数大于1的索引）
    collision_item_groups = [index2id[index] for index in index2id if len(index2id[index]) > 1]
    return collision_item_groups


def _parse_datasets_arg(single: str | None, multiple: list[str] | None) -> list[str]:
    if multiple:
        return [d.strip() for d in multiple if str(d).strip()]
    if not single:
        return []
    return [d.strip() for d in str(single).split(",") if d.strip()]


def _ensure_prefix(num_layers: int) -> list[str]:
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]
    if num_layers > len(prefix):
        raise ValueError(
            f"num_layers={num_layers} exceeds supported prefix list size={len(prefix)}. "
            "Please extend the prefix list in generate_indices.py."
        )
    return prefix[:num_layers]


def _build_key(indices_row: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in indices_row.tolist())


def _key_to_tokens(key: tuple[int, ...], prefix: list[str]) -> list[str]:
    return [prefix[i].format(int(ind)) for i, ind in enumerate(key)]


def _dataset_spans(datasets: list[str], data: MultiEmbDataset) -> list[tuple[str, int, int]]:
    if len(datasets) != len(data.embeddings_list):
        raise ValueError(
            f"--datasets length ({len(datasets)}) must match data_paths length ({len(data.embeddings_list)})."
        )
    spans: list[tuple[str, int, int]] = []
    start = 0
    for ds, emb in zip(datasets, data.embeddings_list, strict=False):
        end = start + len(emb)
        spans.append((ds, start, end))
        start = end
    return spans


def _dataset_ids_for_global_indices(spans: list[tuple[str, int, int]], n: int) -> list[int]:
    dataset_ids = [-1] * n
    for ds_id, (_ds, start, end) in enumerate(spans):
        for i in range(start, end):
            dataset_ids[i] = ds_id
    return dataset_ids


def _get_attr(obj, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _as_int_list(value) -> list[int]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    out: list[int] = []
    for item in values:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def _unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _infer_dataset_name_from_path(path: str) -> str | None:
    if not path:
        return None
    parent = os.path.basename(os.path.dirname(os.path.normpath(path)))
    return parent or None


def _infer_embedding_model_from_path(path: str) -> str | None:
    base = os.path.basename(path)
    match = re.search(r"\.emb-(.+?)-td\.npy$", base)
    if match:
        return match.group(1)
    match = re.search(r"\.emb-(.+?)\.npy$", base)
    if match:
        return match.group(1)
    return None


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9-]+", "-", text).strip("-")
    return cleaned or "na"


def _extract_train_data_paths(model_args) -> list[str]:
    data_paths = _as_str_list(_get_attr(model_args, "data_paths", None))
    if data_paths:
        return data_paths
    return _as_str_list(_get_attr(model_args, "data_path", None))


def _build_auto_output_suffix(ckpt_path: str, model_args, fallback_data_paths: list[str]) -> str:
    train_data_paths = _extract_train_data_paths(model_args)
    if not train_data_paths:
        train_data_paths = list(fallback_data_paths)

    train_datasets = _unique_keep_order(
        [ds for ds in (_infer_dataset_name_from_path(path) for path in train_data_paths) if ds]
    )
    train_embeddings = _unique_keep_order(
        [emb for emb in (_infer_embedding_model_from_path(path) for path in train_data_paths) if emb]
    )

    num_emb_list = _as_int_list(_get_attr(model_args, "num_emb_list", None))
    rq_layers = len(num_emb_list)
    cb_tag = "-".join(str(x) for x in num_emb_list) if num_emb_list else "na"

    embedding_tag = "-".join(_slug(x) for x in train_embeddings) if train_embeddings else "unknown-emb"
    dataset_tag = "-".join(_slug(x) for x in train_datasets) if train_datasets else "unknown-ds"

    run_id = _slug(os.path.basename(os.path.dirname(os.path.abspath(ckpt_path))))
    if run_id == "na":
        run_id = _slug(os.path.splitext(os.path.basename(ckpt_path))[0])

    suffix = f".index_emb-{embedding_tag}_rq{rq_layers}_cb{_slug(cb_tag)}_ds{dataset_tag}_rid{run_id}.json"
    return suffix


def _infer_datasets_from_data_paths(data_paths: list[str]) -> list[str]:
    datasets: list[str] = []
    for path in data_paths:
        ds = _infer_dataset_name_from_path(path)
        if not ds:
            raise ValueError("Cannot infer dataset name from data_path. Please pass --dataset/--datasets explicitly.")
        datasets.append(ds)
    return datasets


def _write_generate_metrics(
    ckpt_path: str,
    datasets: list[str],
    multi_output: bool,
    output_suffix: str,
    max_reencode_rounds: int,
    reencode_rounds: int,
    total_items: int,
    unique_indices: int,
    collision_rate: float,
    max_conflicts: int,
    cross_dataset_collision_groups_round0: int | None,
):
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ckpt_path": os.path.abspath(ckpt_path),
        "datasets": datasets,
        "multi_output": multi_output,
        "output_suffix": output_suffix,
        "max_reencode_rounds": int(max_reencode_rounds),
        "reencode_rounds": int(reencode_rounds),
        "total_items": int(total_items),
        "unique_indices": int(unique_indices),
        "collision_rate": float(collision_rate),
        "max_conflicts": int(max_conflicts),
        "cross_dataset_collision_groups_round0": (
            int(cross_dataset_collision_groups_round0) if cross_dataset_collision_groups_round0 is not None else None
        ),
    }

    rid = _slug(os.path.basename(ckpt_dir))
    metrics_file = os.path.join(
        ckpt_dir,
        f"generate_metrics_{rid}.json",
    )

    with open(metrics_file, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"[metrics] {metrics_file}")


def main(args):
    device = torch.device(args.device)

    # 加载模型检查点
    ckpt = torch.load(args.ckpt_path, map_location=torch.device("cpu"), weights_only=False)  # 加载检查点到 CPU
    model_args = ckpt["args"]  # 从检查点中获取训练参数
    state_dict = ckpt["state_dict"]  # 从检查点中获取模型状态字典

    # 加载嵌入数据集：优先使用命令行传入的 data_path(s)，否则使用 checkpoint 中保存的 data_path(s)
    if getattr(args, "data_paths", None) is not None and getattr(args, "data_path", None) is not None:
        raise ValueError("Please use either --data_path or --data_paths, not both.")

    if getattr(args, "data_paths", None) is not None:
        data_paths = list(args.data_paths)
    elif getattr(args, "data_path", None) is not None:
        data_paths = [args.data_path]
    else:
        data_paths = _extract_train_data_paths(model_args)

    if not data_paths:
        raise ValueError(
            "Cannot infer input data_path(s). Please pass --data_path/--data_paths "
            "or use a checkpoint with saved training data path(s)."
        )

    datasets = _parse_datasets_arg(getattr(args, "dataset", None), getattr(args, "datasets", None))
    if not datasets:
        datasets = _infer_datasets_from_data_paths(data_paths)

    multi_output = len(datasets) > 1
    if multi_output and getattr(args, "output_file", None) is not None:
        raise ValueError("In multi-output mode, please use --output_suffix, not --output_file.")

    if multi_output and len(data_paths) != len(datasets):
        raise ValueError(
            f"Multi-output mode requires one data_path per dataset. "
            f"Got datasets={len(datasets)}, data_paths={len(data_paths)}."
        )

    output_suffix = args.output_suffix
    if not output_suffix:
        output_suffix = _build_auto_output_suffix(args.ckpt_path, model_args, data_paths)
        print(f"[auto_name] output_suffix={output_suffix}")

    data = MultiEmbDataset(data_paths) if len(data_paths) > 1 else EmbDataset(data_paths[0])  # 加载嵌入数据集

    # 初始化 RQVAE 模型
    model = RQVAE(
        in_dim=data.dim,  # 输入维度
        num_emb_list=model_args.num_emb_list,  # 每层量化器的嵌入数量列表
        e_dim=model_args.e_dim,  # 嵌入维度
        layers=model_args.layers,  # RQVAE 层数
        dropout_prob=model_args.dropout_prob,  # Dropout 概率
        bn=model_args.bn,  # 是否使用 Batch Normalization
        loss_type=model_args.loss_type,  # 损失类型
        quant_loss_weight=model_args.quant_loss_weight,  # 量化损失权重
        kmeans_init=model_args.kmeans_init,  # 是否使用 KMeans 初始化
        kmeans_iters=model_args.kmeans_iters,  # KMeans 迭代次数
        sk_epsilons=model_args.sk_epsilons,  # Sinkhorn-Knopp 算法的 epsilon 值列表
        sk_iters=model_args.sk_iters,  # Sinkhorn-Knopp 算法的迭代次数
    )

    model.load_state_dict(state_dict)  # 加载模型状态字典
    model = model.to(device)  # 将模型移动到指定设备
    model.eval()  # 设置模型为评估模式
    print(model)  # 打印模型结构

    # 创建 DataLoader
    data_loader = DataLoader(
        data,
        num_workers=model_args.num_workers,  # 工作进程数
        batch_size=args.batch_size,  # 批处理大小
        shuffle=False,  # 不打乱数据
        pin_memory=True,  # 启用 pin_memory，加快数据传输到 GPU
    )

    prefix = _ensure_prefix(len(model_args.num_emb_list))

    all_indices = []  # 存储所有索引（int）的列表
    all_keys: list[tuple[int, ...]] = []  # 用于碰撞检查的 key（tuple）

    # 遍历数据加载器，生成初始索引
    for d in tqdm(data_loader):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)  # 获取模型索引，不使用 Sinkhorn-Knopp
        # 将索引展平并转换为 numpy 数组
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            key = _build_key(index)
            all_indices.append(index.tolist())
            all_keys.append(key)

    all_indices = np.asarray(all_indices, dtype=np.int64)

    # 复用 checkpoint 内保存的 sk_epsilon 配置，不在导出阶段覆盖。

    tt = 0  # 迭代计数器
    # There are often duplicate items in the dataset, and we no longer differentiate them
    # 循环处理碰撞，直到没有碰撞或达到最大迭代次数
    spans = None
    dataset_ids = None
    if multi_output:
        assert isinstance(data, MultiEmbDataset)
        spans = _dataset_spans(datasets, data)
        dataset_ids = _dataset_ids_for_global_indices(spans, len(all_keys))

    cross_dataset_collision_groups_round0 = None

    while True:
        if tt >= args.max_reencode_rounds or check_collision(all_keys):
            break

        collision_item_groups = get_collision_item(all_keys)  # 获取碰撞的项目组
        if tt == 0 and dataset_ids is not None:
            cross = 0
            for g in collision_item_groups:
                ds_set = {dataset_ids[i] for i in g if dataset_ids[i] >= 0}
                if len(ds_set) > 1:
                    cross += 1
            cross_dataset_collision_groups_round0 = cross
            print(
                f"[collision] groups={len(collision_item_groups)} "
                f"(cross-dataset={cross}, max_rounds={args.max_reencode_rounds})"
            )

        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)  # 获取发生碰撞的数据

            indices = model.get_indices(d, use_sk=True)  # 使用 Sinkhorn-Knopp 算法重新获取索引
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices, strict=False):  # 遍历碰撞项目和新的索引
                all_indices[item] = index
                all_keys[item] = _build_key(index)
        tt += 1

    print("All indices number: ", len(all_indices))  # 打印总索引数量
    print(
        "Max number of conflicts: ",
        max(get_indices_count(all_keys).values()),  # 打印最大冲突数量
    )

    tot_item = len(all_keys)
    tot_indice = len(set(all_keys))
    collision_rate = (tot_item - tot_indice) / tot_item
    print("Collision Rate", collision_rate)  # 打印最终碰撞率

    if multi_output:
        assert spans is not None
        os.makedirs(args.output_dir, exist_ok=True)
        for ds, start, end in spans:
            out_dir = os.path.join(args.output_dir, ds)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{ds}{output_suffix}")

            ds_dict = {}
            for local_id, key in enumerate(all_keys[start:end]):
                ds_dict[local_id] = _key_to_tokens(key, prefix)

            with open(out_path, "w", encoding="utf-8") as fp:
                json.dump(ds_dict, fp)
            print(f"[output] {ds}: {out_path} (items={end - start})")
    else:
        dataset_name = datasets[0]
        output_file = args.output_file or f"{dataset_name}{output_suffix}"

        all_indices_dict = {}  # 字典，用于存储最终的索引映射
        for item, key in enumerate(all_keys):
            all_indices_dict[item] = _key_to_tokens(key, prefix)

        os.makedirs(args.output_dir, exist_ok=True)  # 确保输出目录存在
        output_file_path = os.path.join(args.output_dir, output_file)
        with open(output_file_path, "w", encoding="utf-8") as fp:
            json.dump(all_indices_dict, fp)
        print(f"[output] {output_file_path} (items={len(all_keys)})")

    _write_generate_metrics(
        ckpt_path=args.ckpt_path,
        datasets=datasets,
        multi_output=multi_output,
        output_suffix=output_suffix,
        max_reencode_rounds=args.max_reencode_rounds,
        reencode_rounds=tt,
        total_items=tot_item,
        unique_indices=tot_indice,
        collision_rate=collision_rate,
        max_conflicts=max(get_indices_count(all_keys).values()),
        cross_dataset_collision_groups_round0=cross_dataset_collision_groups_round0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate indices for Multimodal Recommendation Model")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (single dataset) or comma-separated names (multi).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Multiple dataset names (space separated).",
    )
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Optional override for embedding .npy path (single dataset).",
    )
    parser.add_argument(
        "--data_paths",
        type=str,
        nargs="+",
        default=None,
        help="Optional override for embedding .npy paths (multiple datasets).",
    )
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help=("Output JSON file name (single-output mode). If omitted, auto-naming is used."),
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default=None,
        help=("Output suffix for auto output naming. Default: auto-generated rich suffix with emb/rq/cb/ds/rid."),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (e.g., cuda:0 or cpu)",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for data loading")
    parser.add_argument(
        "--max_reencode_rounds",
        type=int,
        default=20,
        help="Max rounds to re-encode collided items using Sinkhorn-Knopp.",
    )

    args = parser.parse_args()
    main(args)
