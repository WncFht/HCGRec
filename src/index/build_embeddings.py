import argparse
import os
import random
import time

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from index.utils import clean_text, load_json


def load_data(args):
    if args.root:
        print("args.root: ", args.root)
    item2feature_path = os.path.join(args.root, f"{args.dataset}.item.json")
    item2feature = load_json(item2feature_path)
    return item2feature


def generate_text(item2feature, features):
    item_text_list = []

    def _sort_key(item_key: str):
        try:
            return (0, int(item_key))
        except Exception:
            return (1, str(item_key))

    for item in sorted(item2feature.keys(), key=_sort_key):
        data = item2feature[item]
        text = []
        for meta_key in features:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                cleaned = meta_value.strip()
                if cleaned != "":
                    text.append(cleaned)

        if len(text) == 0:
            text = ["unknown item"]

        try:
            item_id = int(item)
        except Exception:
            item_id = item

        item_text_list.append((item_id, " ".join(text)))

    return item_text_list


def preprocess_text(args):
    print("Process text data: ")
    print("Dataset: ", args.dataset)
    item2feature = load_data(args)
    item_text_list = generate_text(item2feature, ["title", "description"])
    return item_text_list


def generate_item_embedding(args, item_text_list, tokenizer, model, accelerator, word_drop_ratio=-1):
    """
    Multi-process embedding extraction without NCCL collectives.

    Why:
    - `accelerator.wait_for_everyone()` uses NCCL barrier and can crash/hang on some kernels
      (e.g., "Failed to CUDA calloc ..." or TCPStore timeout).
    - For embedding extraction, we only need a final concatenated `.npy`, so we can do a
      file-based gather: each rank writes a part file, then rank0 merges them.
    """
    all_ids, all_texts = zip(*item_text_list, strict=False)
    total_items = len(all_texts)

    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    chunk_size = int(np.ceil(total_items / num_processes))
    start_idx = process_index * chunk_size
    end_idx = min(start_idx + chunk_size, total_items)

    local_ids = all_ids[start_idx:end_idx]
    local_texts = all_texts[start_idx:end_idx]

    if accelerator.is_main_process:
        print(f"Total items: {total_items}")
        print(f"Start generating embeddings with {num_processes} processes...")

    batch_size = int(args.batch_size)

    pbar = tqdm(
        total=len(local_texts),
        desc=f"Proc {process_index}",
        disable=not accelerator.is_local_main_process,
    )

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _mean_pool_last_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor):
        mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)  # [B, S, 1]
        sum_embeddings = torch.sum(last_hidden * mask, dim=1, dtype=torch.float32)  # [B, D]
        sum_mask = torch.clamp(mask.sum(dim=1, dtype=torch.float32), min=1e-9)  # [B, 1]
        return sum_embeddings / sum_mask

    final_prefix = os.path.join(args.root, f"{args.dataset}.emb-{args.plm_name}-td")
    part_dir = getattr(args, "tmp_dir", None) or args.root
    out_prefix = os.path.join(part_dir, f"{args.dataset}.emb-{args.plm_name}-td")
    run_id = getattr(args, "run_id", None) or "default"

    part_emb_path = f"{out_prefix}.{run_id}.part{process_index}.npy"
    part_ids_path = f"{out_prefix}.{run_id}.part{process_index}.ids.json"
    part_emb_tmp = f"{part_emb_path}.tmp"
    part_ids_tmp = f"{part_ids_path}.tmp"

    # Clean leftovers for this rank
    for p in (part_emb_path, part_ids_path, part_emb_tmp, part_ids_tmp):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    local_ids_out: list = []
    emb_fp = None
    rows_written = 0
    emb_dim = None

    def _write_npy_header(fp, shape, dtype=np.float32):
        header = {
            "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
            "fortran_order": False,
            "shape": shape,
        }
        write_header = getattr(
            np.lib.format,
            "write_array_header_2_0",
            np.lib.format.write_array_header_1_0,
        )
        write_header(fp, header)

    with torch.inference_mode():
        i = 0
        while i < len(local_texts):
            cur_bs = min(batch_size, len(local_texts) - i)
            batch_texts = list(local_texts[i : i + cur_bs])
            batch_ids = local_ids[i : i + cur_bs]

            if word_drop_ratio > 0:
                processed_batch = []
                for text in batch_texts:
                    sent = text.split(" ")
                    new_sent = [wd for wd in sent if random.random() > word_drop_ratio]
                    processed_batch.append(" ".join(new_sent))
                batch_texts = processed_batch

            try:
                encoded_sentences = tokenizer(
                    batch_texts,
                    max_length=args.max_sent_len,
                    truncation=True,
                    return_tensors="pt",
                    padding=True,
                ).to(accelerator.device)

                input_ids = encoded_sentences.input_ids
                attention_mask = encoded_sentences.attention_mask

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden = outputs.last_hidden_state
                mean_output = _mean_pool_last_hidden(last_hidden, attention_mask)
            except torch.cuda.OutOfMemoryError:
                if accelerator.is_local_main_process:
                    print(
                        f"[OOM] batch_size={batch_size}, max_len={args.max_sent_len}. "
                        "Reducing batch_size and retrying..."
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                continue

            mean_np = mean_output.cpu().numpy().astype(np.float32, copy=False)
            if not mean_np.flags["C_CONTIGUOUS"]:
                mean_np = np.ascontiguousarray(mean_np)

            if emb_fp is None:
                emb_dim = int(mean_np.shape[1])
                os.makedirs(os.path.dirname(part_emb_tmp) or ".", exist_ok=True)
                emb_fp = open(part_emb_tmp, "wb")
                _write_npy_header(emb_fp, (len(local_texts), emb_dim), dtype=np.float32)

            emb_fp.write(mean_np.tobytes(order="C"))
            rows_written += int(mean_np.shape[0])
            local_ids_out.extend(batch_ids)

            pbar.update(len(batch_texts))
            i += cur_bs

    pbar.close()

    if emb_fp is None:
        raise ValueError("No embeddings were generated; please check input text data.")
    emb_fp.close()

    if rows_written != len(local_texts):
        raise RuntimeError(
            f"Embedding write mismatch on rank {process_index}: wrote {rows_written} rows "
            f"but expected {len(local_texts)}"
        )
    if len(local_ids_out) != len(local_texts):
        raise RuntimeError(
            f"IDs length mismatch on rank {process_index}: len(ids)={len(local_ids_out)} "
            f"but expected {len(local_texts)}"
        )

    os.replace(part_emb_tmp, part_emb_path)
    with open(part_ids_tmp, "w", encoding="utf-8") as f:
        import json

        json.dump(local_ids_out, f, ensure_ascii=False)
    os.replace(part_ids_tmp, part_ids_path)

    print(
        f"[rank{process_index}] Saved part files: {part_emb_path} and {part_ids_path} (n={len(local_texts)})",
        flush=True,
    )

    if not accelerator.is_main_process:
        return

    # Merge on rank0 (no distributed barrier)
    final_emb_path = f"{final_prefix}.npy"
    final_ids_path = f"{final_prefix}.ids.json"
    final_emb_tmp = f"{final_emb_path}.tmp"
    final_ids_tmp = f"{final_ids_path}.tmp"

    def _expected_count(rank: int) -> int:
        s = rank * chunk_size
        e = min(s + chunk_size, total_items)
        return max(0, e - s)

    expected_counts = [_expected_count(r) for r in range(num_processes)]
    if sum(expected_counts) != total_items:
        raise RuntimeError("Unexpected chunk sizing; please check process partition logic.")

    print("Rank0 waiting for all part files...", flush=True)
    missing = set(range(num_processes))
    start_wait = time.time()
    last_report = 0.0
    last_missing = None
    while missing:
        done = []
        for r in missing:
            p_emb = f"{out_prefix}.{run_id}.part{r}.npy"
            p_ids = f"{out_prefix}.{run_id}.part{r}.ids.json"
            if os.path.exists(p_emb) and os.path.exists(p_ids):
                done.append(r)
        for r in done:
            missing.remove(r)

        if missing:
            now = time.time()
            if last_missing != missing or now - last_report > 60:
                waited = int(now - start_wait)
                print(
                    f"Waiting for part files from ranks: {sorted(missing)} (waited {waited}s)",
                    flush=True,
                )
                last_report = now
                last_missing = set(missing)
            if time.time() - start_wait > 6 * 3600:
                raise TimeoutError(f"Timed out waiting for part files from ranks: {sorted(missing)}")
            time.sleep(2)

    part0 = np.load(f"{out_prefix}.{run_id}.part0.npy")
    part0 = np.squeeze(part0)
    if part0.ndim != 2:
        raise ValueError(f"Unexpected part0 shape: {part0.shape}")
    dim = int(part0.shape[1])
    del part0

    os.makedirs(os.path.dirname(final_emb_tmp) or ".", exist_ok=True)
    final_fp = open(final_emb_tmp, "wb")
    _write_npy_header(final_fp, (total_items, dim), dtype=np.float32)
    all_item_ids: list = []

    offset = 0
    for r in range(num_processes):
        n_r = expected_counts[r]
        print(f"Merging part {r}/{num_processes - 1} (rows={n_r})...", flush=True)
        p_emb = np.load(f"{out_prefix}.{run_id}.part{r}.npy")
        p_emb = np.squeeze(p_emb)
        if p_emb.ndim != 2 or int(p_emb.shape[0]) != n_r or int(p_emb.shape[1]) != dim:
            raise ValueError(f"Part {r} shape mismatch: got {p_emb.shape}, expected ({n_r}, {dim})")
        if p_emb.dtype != np.float32:
            p_emb = p_emb.astype(np.float32, copy=False)
        if not p_emb.flags["C_CONTIGUOUS"]:
            p_emb = np.ascontiguousarray(p_emb)
        final_fp.write(p_emb.tobytes(order="C"))

        with open(f"{out_prefix}.{run_id}.part{r}.ids.json", encoding="utf-8") as f:
            import json

            ids_r = json.load(f)
        if len(ids_r) != n_r:
            raise ValueError(f"Part {r} ids length mismatch: len={len(ids_r)} expected={n_r}")
        all_item_ids.extend(ids_r)
        offset += n_r

    final_fp.close()

    os.replace(final_emb_tmp, final_emb_path)
    with open(final_ids_tmp, "w", encoding="utf-8") as f:
        import json

        json.dump(all_item_ids, f, ensure_ascii=False)
    os.replace(final_ids_tmp, final_ids_path)

    print("Merge finished. Saving done.")
    print(f"Final Embeddings shape: ({len(all_item_ids)}, {dim})")
    print(f"Saved to {final_emb_path}")
    print(f"Saved item ids to {final_ids_path}")

    for r in range(num_processes):
        for p in (
            f"{out_prefix}.{run_id}.part{r}.npy",
            f"{out_prefix}.{run_id}.part{r}.ids.json",
        ):
            try:
                os.remove(p)
            except OSError:
                pass


def load_qwen_model(model_path):
    print("Loading Qwen Model:", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    return tokenizer, model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Beauty", help="Beauty / Sports / Toys")
    parser.add_argument("--root", type=str, default="")
    # parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument("--plm_name", type=str, default="qwen")
    parser.add_argument("--plm_checkpoint", type=str, default="xxx", help="Qwen model path")
    parser.add_argument("--max_sent_len", type=int, default=2048)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Inference batch size per process/GPU. Reduce if you hit OOM.",
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default=None,
        help=(
            "Optional directory for temporary part files (recommended on network FS, e.g. /tmp). Defaults to --root."
        ),
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Unique run id for temp part files (recommended when num_processes>1).",
    )
    parser.add_argument("--word_drop_ratio", type=float, default=-1, help="word drop ratio")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    accelerator = Accelerator()

    if accelerator.is_main_process:
        print(f"Running with {accelerator.num_processes} processes.")

    item_text_list = preprocess_text(args)

    plm_tokenizer, plm_model = load_qwen_model(args.plm_checkpoint)

    plm_model = plm_model.to(accelerator.device)
    plm_model.eval()

    generate_item_embedding(
        args,
        item_text_list,
        plm_tokenizer,
        plm_model,
        accelerator,
        word_drop_ratio=args.word_drop_ratio,
    )
