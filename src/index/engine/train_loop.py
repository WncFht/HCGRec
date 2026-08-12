from time import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from index.utils import set_color


class TrainLoopEngine:
    def __init__(self, trainer):
        self.trainer = trainer

    @property
    def model(self):
        return self.trainer.model

    @property
    def device(self):
        return self.trainer.device

    @property
    def logger(self):
        return self.trainer.logger

    @property
    def is_main_process(self):
        return self.trainer.is_main_process

    @property
    def distributed(self):
        return self.trainer.distributed

    @property
    def world_size(self):
        return self.trainer.world_size

    @property
    def optimizer(self):
        return self.trainer.optimizer

    @property
    def scheduler(self):
        return self.trainer.scheduler

    @property
    def epochs(self):
        return self.trainer.epochs

    @property
    def eval_step(self):
        return self.trainer.eval_step

    @property
    def use_wandb(self):
        return self.trainer.use_wandb

    @property
    def wandb(self):
        return self.trainer.wandb

    def train_epoch(self, train_data, epoch_idx):
        self.model.train()
        base_model = self.trainer._unwrap_model()

        total_loss = 0.0
        total_recon_loss = 0.0
        iter_data = tqdm(
            train_data,
            total=len(train_data),
            ncols=100,
            desc=set_color(f"Train {epoch_idx}", "pink"),
            disable=not self.is_main_process,
        )
        for data in iter_data:
            data = data.to(self.device)

            if not torch.isfinite(data).all():
                flat_finite = torch.isfinite(data).view(data.size(0), -1)
                finite_mask = flat_finite.all(dim=1)
                num_bad = int((~finite_mask).sum().item())
                if num_bad > 0 and self.is_main_process:
                    self.logger.warning(f"Train epoch {epoch_idx}: dropping {num_bad} samples with NaN/Inf.")
                data = data[finite_mask]
                if data.size(0) == 0:
                    continue

            self.optimizer.zero_grad()
            out, rq_loss, _indices = self.model(data)
            loss, loss_recon = base_model.compute_loss(out, rq_loss, xs=data)
            self.trainer._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((self.model.parameters()), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            if self.use_wandb:
                self.wandb.log(
                    {
                        "train_step/loss": loss.item(),
                        "train_step/recon_loss": loss_recon.item(),
                        "train_step/lr": self.scheduler.get_last_lr()[0],
                    }
                )

            total_loss += loss.item()
            total_recon_loss += loss_recon.item()

        return total_loss, total_recon_loss

    def maybe_run_large_scale_kmeans_init(self, train_data):
        if not getattr(self.trainer.args, "kmeans_init", False) or not getattr(
            self.trainer.args, "large_scale_kmeans", False
        ):
            return

        base_model = self.trainer._unwrap_model()
        need_init = not all(vq.initted for vq in base_model.rq.vq_layers)
        if not need_init:
            return

        if self.distributed and not self.is_main_process:
            self.logger.info("Waiting for LARGE SCALE K-Means initialization on rank 0...")
        self.logger.info("Performing LARGE SCALE K-Means initialization...")

        did_init = False
        if self.is_main_process:
            dataset = getattr(train_data, "dataset", train_data)
            init_size = min(50000, len(dataset))
            if init_size <= 0:
                self.logger.warning("Skip LARGE SCALE K-Means initialization: empty dataset.")
                init_size = 0

            init_data = None
            init_indices = (
                np.random.choice(len(dataset), size=init_size, replace=False)
                if init_size > 0
                else np.array([], dtype=np.int64)
            )
            if hasattr(dataset, "embeddings"):
                init_data_tensors = np.array(
                    dataset.embeddings[init_indices],
                    dtype=np.float32,
                    copy=True,
                )
                init_data = torch.from_numpy(init_data_tensors).to(self.device)
            else:
                try:
                    init_batch = dataset[init_indices]
                    if isinstance(init_batch, torch.Tensor):
                        init_data = init_batch.to(self.device)
                    else:
                        init_data = torch.as_tensor(
                            init_batch,
                            dtype=torch.float32,
                            device=self.device,
                        )
                except Exception as err:
                    collected = []
                    collected_num = 0
                    for batch in train_data:
                        batch_tensor = (
                            batch if isinstance(batch, torch.Tensor) else torch.as_tensor(batch, dtype=torch.float32)
                        )
                        collected.append(batch_tensor)
                        collected_num += batch_tensor.shape[0]
                        if collected_num >= init_size:
                            break
                    if not collected:
                        raise ValueError("Failed to collect init data for K-Means initialization.") from err
                    init_data = torch.cat(collected, dim=0)[:init_size].to(self.device)

            if init_data is not None and init_data.numel() > 0:
                flat_finite = torch.isfinite(init_data).view(init_data.size(0), -1)
                finite_mask = flat_finite.all(dim=1)
                num_bad = int((~finite_mask).sum().item())
                if num_bad > 0:
                    self.logger.warning(f"LARGE SCALE K-Means init: dropping {num_bad} samples with NaN/Inf.")
                    init_data = init_data[finite_mask]

            if init_data is None or init_data.numel() == 0:
                self.logger.warning("Skip LARGE SCALE K-Means initialization: empty init_data.")
            else:
                with torch.no_grad():
                    self.model(init_data)
                did_init = True

        if self.distributed:
            did_init_tensor = torch.tensor(
                [1 if did_init else 0],
                device=self.device,
                dtype=torch.int64,
            )
            dist.broadcast(did_init_tensor, src=0)
            did_init = bool(did_init_tensor.item())

        if self.distributed and did_init:
            self.trainer._broadcast_vq_codebooks()

        self.logger.info("LARGE SCALE K-Means initialization finished.")

    def run_fit(self, train_loader):
        self.maybe_run_large_scale_kmeans_init(train_loader)

        eval_loader = None

        for epoch_idx in range(self.epochs):
            if self.distributed and hasattr(getattr(train_loader, "sampler", None), "set_epoch"):
                train_loader.sampler.set_epoch(epoch_idx)

            training_start_time = time()
            train_loss, train_recon_loss = self.train_epoch(train_loader, epoch_idx)
            training_end_time = time()

            if self.distributed:
                loss_tensor = torch.tensor(
                    [train_loss, train_recon_loss],
                    device=self.device,
                    dtype=torch.float64,
                )
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                train_loss, train_recon_loss = loss_tensor.tolist()

            train_loss_output = self.trainer._generate_train_loss_output(
                epoch_idx,
                training_start_time,
                training_end_time,
                train_loss,
                train_recon_loss,
            )
            if self.is_main_process:
                self.logger.info(train_loss_output)

            if (epoch_idx + 1) % self.eval_step != 0:
                continue

            if self.distributed:
                dist.barrier()

            if self.is_main_process:
                valid_start_time = time()
                if eval_loader is None:
                    dataset = getattr(train_loader, "dataset", train_loader)
                    eval_loader = DataLoader(
                        dataset,
                        num_workers=getattr(self.trainer.args, "num_workers", 4),
                        batch_size=getattr(self.trainer.args, "batch_size", 2048),
                        shuffle=False,
                        pin_memory=True,
                    )
                collision_rate, avg_utilization = self.trainer.eval_engine.valid_epoch(eval_loader)

                if train_loss < self.trainer.best_loss:
                    self.trainer.best_loss = train_loss
                    self.trainer.checkpoint_manager.save_checkpoint(
                        epoch=epoch_idx,
                        ckpt_file=self.trainer.best_loss_ckpt,
                    )

                if collision_rate < self.trainer.best_collision_rate:
                    self.trainer.best_collision_rate = collision_rate
                    self.trainer.checkpoint_manager.save_checkpoint(
                        epoch_idx,
                        collision_rate=collision_rate,
                        avg_utilization=avg_utilization,
                        ckpt_file=self.trainer.best_collision_ckpt,
                    )

                if avg_utilization > self.trainer.best_codebook_utilization:
                    self.trainer.best_codebook_utilization = avg_utilization
                    self.trainer.checkpoint_manager.save_checkpoint(
                        epoch=epoch_idx,
                        collision_rate=collision_rate,
                        avg_utilization=avg_utilization,
                        ckpt_file=self.trainer.best_utilization_ckpt,
                    )

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %.4f, "
                    + set_color("avg_utilization", "blue")
                    + ": %.4f]"
                ) % (
                    epoch_idx,
                    valid_end_time - valid_start_time,
                    collision_rate,
                    avg_utilization,
                )
                self.logger.info(valid_score_output)

                if self.use_wandb:
                    denom = len(train_loader) * self.world_size
                    self.wandb.log(
                        {
                            "epoch": epoch_idx,
                            "epoch/train_loss": train_loss / denom,
                            "epoch/train_recon_loss": train_recon_loss / denom,
                            "eval/collision_rate": collision_rate,
                            "eval/avg_codebook_utilization": avg_utilization,
                            "eval/best_loss": self.trainer.best_loss,
                            "eval/best_collision_rate": self.trainer.best_collision_rate,
                            "eval/best_codebook_utilization": self.trainer.best_codebook_utilization,
                        }
                    )

                self.trainer.checkpoint_manager.update_and_prune_checkpoints(
                    epoch_idx,
                    collision_rate=collision_rate,
                    avg_utilization=avg_utilization,
                )

            if self.distributed:
                dist.barrier()

        if self.use_wandb:
            self.wandb.finish()

        return self.trainer.best_loss, self.trainer.best_collision_rate
