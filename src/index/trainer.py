import logging

import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from transformers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from index.engine import CheckpointManager, EvalEngine, TrainLoopEngine


# wandb 变成可选依赖，没安装也能跑
try:
    import wandb  # type: ignore
except ImportError:
    wandb = None


class Trainer:
    """训练器类，用于管理模型的训练、评估和保存。"""

    def __init__(self, args, model, data_num):
        """初始化训练器。

        参数:
            args: 包含各种配置参数的对象。
            model: 要训练的模型实例。
            data_num: 训练数据集中的样本总数（以 step 数为单位，比如 len(dataloader)）。
        """
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.is_main_process = self.rank == 0

        if self.distributed and not self.is_main_process:
            self.logger.setLevel(logging.WARNING)

        want_wandb = getattr(self.args, "use_wandb", False)
        self.use_wandb = bool(want_wandb and self.is_main_process and wandb is not None)
        if want_wandb and wandb is None and self.is_main_process:
            self.logger.warning("use_wandb=True 但未安装 wandb，自动关闭 wandb 日志。")

        if self.use_wandb:
            wandb_project = getattr(self.args, "wandb_project", "grec_index")
            wandb_name = getattr(self.args, "wandb_name", None)
            wandb.init(
                project=wandb_project,
                config=self.args,
                reinit=True,
                name=wandb_name,
            )
            wandb.watch(self.model, log="all", log_freq=max(100, data_num // 10))

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num

        self.save_limit = args.save_limit
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = torch.device(args.device)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.best_codebook_utilization = 0.0
        self.best_utilization_ckpt = "best_utilization_model.pth"

        self.checkpoint_manager = CheckpointManager(self)
        self.checkpoint_manager.setup_checkpoint_dir(args.ckpt_dir)
        self.checkpoint_manager.write_run_meta()

        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)

        self.eval_engine = EvalEngine(self)
        self.train_loop = TrainLoopEngine(self)

    @property
    def wandb(self):
        return wandb

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _broadcast_vq_codebooks(self):
        if not self.distributed:
            return
        model = self._unwrap_model()
        for vq in model.rq.vq_layers:
            dist.broadcast(vq.embedding.weight.data, src=0)
            vq.initted = True

    def _build_optimizer(self):
        params = self.model.parameters()
        learner = self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adamw":
            optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            self.logger.warning("Received unrecognized optimizer, set default Adam optimizer")
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps,
            )
        else:
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=self.warmup_steps
            )

        return lr_scheduler

    def _check_nan(self, loss: torch.Tensor):
        if not torch.isfinite(loss):
            raise ValueError("Training loss is NaN or Inf")

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        from index.utils import set_color

        train_loss_output = (
            set_color("epoch %d training", "green") + " [" + set_color("time", "blue") + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + f": {loss:.4f}"
        train_loss_output += ", "
        train_loss_output += set_color("reconstruction loss", "blue") + f": {recon_loss:.4f}"
        return train_loss_output + "]"

    def fit(self, data):
        return self.train_loop.run_fit(data)
