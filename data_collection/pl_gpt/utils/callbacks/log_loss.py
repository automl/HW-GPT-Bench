from typing import Dict
import pathlib
import pytorch_lightning as pl
import torch
import numpy as np
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only


class LogLoss(Callback):

    def __init__(
        self,
        log_every_n_steps: int = 1,
        log_quantiles: bool = False,
        dirpath: str = None,
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.dirpath = dirpath
        self.log_quantiles = log_quantiles

        if self.dirpath is not None:
            self.dirpath = pathlib.Path(dirpath) / "unreduced_loss"
            self.dirpath.mkdir(parents=True, exist_ok=True)

            self.loss_list = []

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Dict,
        batch,
        batch_idx,
    ):

        if trainer.global_step % self.log_every_n_steps == 0:

            unreduced_loss = outputs["unreduced_loss"]
            local_rank = trainer.global_rank

            stats = {}

            stats[f"loss_{local_rank}/mean"] = unreduced_loss.mean().item()
            stats[f"loss_{local_rank}/std"] = unreduced_loss.std().item()
            stats[f"loss_{local_rank}/min"] = unreduced_loss.min().item()
            stats[f"loss_{local_rank}/max"] = unreduced_loss.max().item()

            if self.log_quantiles:
                q = (
                    torch.arange(0.25, 1, 0.25)
                    .round(decimals=2)
                    .to(trainer.model.device)
                )
                deciles = torch.quantile(unreduced_loss, q, interpolation="linear")
                for q_idx, d_val in enumerate(deciles):
                    stats[f"loss_{local_rank}/quantile-{q[q_idx]}"] = d_val.item()

            if trainer.loggers is not None:
                for logger in trainer.loggers:
                    logger.log_metrics(stats, step=trainer.global_step)

            if self.dirpath is not None:
                self.loss_list.append(unreduced_loss)

    def on_validation_start(self, trainer, pl_module):

        if self.dirpath is not None and trainer.global_step > 0:
            torch.save(
                self.loss_list,
                self.dirpath
                / f"loss_rank-{trainer.local_rank}_step-{trainer.global_step}.pt",
            )
            self.loss_list = []
