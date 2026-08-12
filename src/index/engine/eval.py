import numpy as np
import torch
from tqdm import tqdm

from index.utils import set_color


class EvalEngine:
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
    def use_wandb(self):
        return self.trainer.use_wandb

    @property
    def wandb(self):
        return self.trainer.wandb

    @torch.no_grad()
    def valid_epoch(self, valid_data):
        self.model.eval()
        base_model = self.trainer._unwrap_model()

        iter_data = tqdm(
            valid_data,
            total=len(valid_data),
            ncols=100,
            desc=set_color("Evaluate   ", "pink"),
            disable=not self.is_main_process,
        )

        all_indices = []
        num_sample = 0
        for data in iter_data:
            data = data.to(self.device)

            if not torch.isfinite(data).all():
                flat_finite = torch.isfinite(data).view(data.size(0), -1)
                finite_mask = flat_finite.all(dim=1)
                num_bad = int((~finite_mask).sum().item())
                if num_bad > 0 and self.is_main_process:
                    self.logger.warning(f"Eval: dropping {num_bad} samples with NaN/Inf.")
                data = data[finite_mask]
                if data.size(0) == 0:
                    continue

            num_sample += data.size(0)
            indices = base_model.get_indices(data)
            indices = indices.view(-1, indices.shape[-1]).cpu()
            all_indices.append(indices)

        if not all_indices or num_sample == 0:
            self.logger.warning(
                "Validation got no valid samples (all NaN/Inf?). Return collision_rate=1.0, avg_utilization=0.0."
            )
            return 1.0, 0.0

        all_indices = torch.cat(all_indices, dim=0).numpy()

        indices_set = set()
        for index in all_indices:
            code = "-".join([str(int(val)) for val in index])
            indices_set.add(code)
        collision_rate = (num_sample - len(indices_set)) / num_sample

        num_quantizers = all_indices.shape[1]
        num_emb_list = base_model.num_emb_list
        utilizations = []
        self.logger.info("Detailed Codebook Utilization:")
        for i in range(num_quantizers):
            unique_codes = np.unique(all_indices[:, i])
            utilization = len(unique_codes) / num_emb_list[i]
            utilizations.append(utilization)
            self.logger.info(f"  Layer {i}: {utilization:.4f} ({len(unique_codes)}/{num_emb_list[i]})")
            if self.use_wandb:
                self.wandb.log({f"eval/codebook_utilization_layer_{i}": utilization})

        avg_utilization = float(np.mean(utilizations))

        return collision_rate, avg_utilization
