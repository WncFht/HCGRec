import heapq
import json
import os
import platform
import socket
import subprocess

import torch

from index.utils import delete_file, ensure_dir, get_local_time, set_color


class CheckpointManager:
    def __init__(self, trainer):
        self.trainer = trainer

    @property
    def model(self):
        return self.trainer.model

    @property
    def logger(self):
        return self.trainer.logger

    @property
    def args(self):
        return self.trainer.args

    @property
    def is_main_process(self):
        return self.trainer.is_main_process

    def setup_checkpoint_dir(self, ckpt_root: str):
        saved_model_dir = f"{get_local_time()}"
        ckpt_dir = os.path.join(ckpt_root, saved_model_dir)
        ensure_dir(ckpt_dir)
        self.trainer.ckpt_dir = ckpt_dir

    def safe_git_output(self, args):
        try:
            out = subprocess.check_output(
                args,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return out or None
        except Exception:
            return None

    def write_run_meta(self):
        if not self.is_main_process:
            return

        args_dict = {}
        if hasattr(self.args, "__dict__"):
            args_dict = dict(vars(self.args))

        train_data_paths = []
        data_paths = args_dict.get("data_paths")
        if isinstance(data_paths, list) and data_paths:
            train_data_paths = [str(path) for path in data_paths]
        else:
            data_path = args_dict.get("data_path")
            if data_path is not None:
                train_data_paths = [str(data_path)]

        meta = {
            "run_name": args_dict.get("run_name"),
            "wandb_name": args_dict.get("wandb_name"),
            "created_at": get_local_time(),
            "ckpt_dir": self.trainer.ckpt_dir,
            "world_size": self.trainer.world_size,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git_commit": self.safe_git_output(["git", "rev-parse", "HEAD"]),
            "git_branch": self.safe_git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "train_data_paths": train_data_paths,
            "args": args_dict,
        }

        run_meta_path = os.path.join(self.trainer.ckpt_dir, "run_meta.json")
        with open(run_meta_path, "w", encoding="utf-8") as file:
            json.dump(meta, file, ensure_ascii=False, indent=2, sort_keys=True)

        self.logger.info(f"Saved run metadata: {run_meta_path}")

    def save_checkpoint(self, epoch, collision_rate=1, avg_utilization=0, ckpt_file=None):
        ckpt_path = (
            os.path.join(self.trainer.ckpt_dir, ckpt_file)
            if ckpt_file
            else os.path.join(
                self.trainer.ckpt_dir,
                f"epoch_{epoch}_collision_{collision_rate:.4f}_util_{avg_utilization:.4f}_model.pth",
            )
        )
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.trainer.best_loss,
            "best_collision_rate": self.trainer.best_collision_rate,
            "best_codebook_utilization": self.trainer.best_codebook_utilization,
            "state_dict": self.trainer._unwrap_model().state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(set_color("Saving current", "blue") + f": {ckpt_path}")

        return ckpt_path

    def update_and_prune_checkpoints(self, epoch_idx, collision_rate, avg_utilization):
        ckpt_path = self.save_checkpoint(
            epoch_idx,
            collision_rate=collision_rate,
            avg_utilization=avg_utilization,
        )
        now_save = (-collision_rate, ckpt_path)
        if len(self.trainer.newest_save_queue) < self.trainer.save_limit:
            self.trainer.newest_save_queue.append(now_save)
            heapq.heappush(self.trainer.best_save_heap, now_save)
            return

        old_save = self.trainer.newest_save_queue.pop(0)
        self.trainer.newest_save_queue.append(now_save)
        if collision_rate < -self.trainer.best_save_heap[0][0]:
            bad_save = heapq.heappop(self.trainer.best_save_heap)
            heapq.heappush(self.trainer.best_save_heap, now_save)

            if bad_save not in self.trainer.newest_save_queue:
                delete_file(bad_save[1])

        if old_save not in self.trainer.best_save_heap:
            delete_file(old_save[1])
