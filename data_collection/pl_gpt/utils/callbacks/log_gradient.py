import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
import deepspeed


class LogParamsAndGrads(Callback):

    def __init__(
        self,
        log_gradient: bool,
        log_params: bool,
        log_quantiles: bool,
        log_every_n_steps: int,
    ):
        super().__init__()
        self.log_gradient = log_gradient
        self.log_params = log_params
        self.log_quantiles = log_quantiles
        self.log_every_n_steps = log_every_n_steps

    # @rank_zero_only
    def on_before_optimizer_step(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer
    ):

        if trainer.global_step % self.log_every_n_steps == 0 and (
            self.log_params or self.log_gradient
        ):

            q = torch.arange(0.25, 1, 0.25).round(decimals=2).to(trainer.model.device)
            stats = {}
            for k, v in pl_module.named_parameters():

                if self.log_params:

                    if trainer.global_rank == 0:

                        v_detached = v.detach()

                        if torch.isnan(v_detached).sum() > 0:
                            print(f"# NaN in param {k}")
                        if torch.isinf(v_detached).sum() > 0:
                            print(f"# Inf in param {k}")

                        stats[f"param/{k}/mean"] = v_detached.mean().item()
                        if v_detached.shape[0] > 1:
                            stats[f"param/{k}/std"] = v_detached.std().item()
                            stats[f"param/{k}/min"] = v_detached.min().item()
                            stats[f"param/{k}/max"] = v_detached.max().item()
                            stats[f"param/{k}/abs_mean"] = (
                                v_detached.abs().mean().item()
                            )
                            stats[f"param/{k}/abs_std"] = v_detached.abs().std().item()
                            stats[f"param/{k}/l2m"] = (v_detached**2).mean().item()
                            stats[f"param/{k}/l2s"] = (v_detached**2).sum().item()

                            if (
                                self.log_quantiles
                                and v_detached.size().numel() < 10000000
                            ):
                                deciles = torch.quantile(
                                    v_detached.float(), q, interpolation="linear"
                                )
                                for q_idx, d_val in enumerate(deciles):
                                    stats[f"param/{k}/quantile-{q[q_idx]}"] = (
                                        d_val.item()
                                    )

                if self.log_gradient and v.requires_grad:

                    if trainer.num_devices > 1:
                        grad_data = deepspeed.utils.safe_get_full_grad(v)
                    else:
                        grad_data = v.grad

                    if grad_data is not None and trainer.global_rank == 0:
                        if torch.isnan(grad_data).sum() > 0:
                            print(f"# NaN in grad {k}")
                        if torch.isinf(grad_data).sum() > 0:
                            print(f"# Inf in grad {k}")

                        if (
                            torch.isnan(grad_data).sum() > 0
                            or torch.isinf(grad_data).sum() > 0
                        ):
                            stats[f"grad/{k}/mean"] = -10
                            if len(grad_data.shape) > 1 or grad_data.shape[0] > 1:
                                stats[f"grad/{k}/std"] = -10
                                stats[f"grad/{k}/min"] = -10
                                stats[f"grad/{k}/max"] = -10
                                stats[f"grad/{k}/abs_mean"] = -10
                                stats[f"grad/{k}/abs_std"] = -10
                                if (
                                    self.log_quantiles
                                    and grad_data.size().numel() < 10000000
                                ):
                                    for q_idx, _ in enumerate(q):
                                        stats[f"param/{k}/quantile-{q[q_idx]}"] = -10

                        stats[f"grad/{k}/mean"] = grad_data.mean().item()
                        if len(grad_data.shape) > 1 or grad_data.shape[0] > 1:
                            stats[f"grad/{k}/std"] = grad_data.std().item()
                            stats[f"grad/{k}/min"] = grad_data.min().item()
                            stats[f"grad/{k}/max"] = grad_data.max().item()
                            stats[f"grad/{k}/abs_mean"] = grad_data.abs().mean().item()
                            stats[f"grad/{k}/mean"] = grad_data.mean().item()
                            stats[f"grad/{k}/abs_std"] = grad_data.abs().std().item()
                            stats[f"grad/{k}/std"] = grad_data.std().item()

                            if (
                                self.log_quantiles
                                and grad_data.size().numel() < 10000000
                            ):
                                deciles = torch.quantile(
                                    grad_data.float(), q, interpolation="linear"
                                )
                                for q_idx, d_val in enumerate(deciles):
                                    stats[f"grad/{k}/quantile-{q[q_idx]}"] = (
                                        d_val.item()
                                    )

            if trainer.loggers is not None and trainer.global_rank == 0:
                for logger in trainer.loggers:
                    logger.log_metrics(stats, step=trainer.global_step)
