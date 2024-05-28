from collections import defaultdict
import pytorch_lightning as pl
import torch
import inspect
from hwgpt.model.gpt.model import GPT
from hwgpt.model.gpt.utils import (
    sample_config_max,
    sample_config_mid,
    sample_config_min,
)
from data_collection.pl_gpt.utils import instantiate
from data_collection.pl_gpt.utils.group_parameters import group_parameters_for_optimizer
from data_collection.pl_gpt.utils.optim.lr_schedule import get_learning_rate_schedule
from typing import Any, Dict


class LanguageModelTrainer(pl.LightningModule):
    """
    PTL wrapper class for model training
    """

    def __init__(
        self,
        cfg_train: Any,
        cfg_model: Any,
        py_logger: Any,
        val_sets_name: Any,
        ignore_index: bool,
        arch_sampled: Dict[str, Any],
    ):
        super().__init__()

        self.save_hyperparameters()
        self.cfg_train = cfg_train
        self.cfg_model = cfg_model

        self.val_sets_name = val_sets_name
        self.ignore_index = ignore_index
        self.py_logger = py_logger

        self.model = GPT(cfg_model)
        checkpoint_path = cfg_model.checkpoint_path
        state_dict = {}
        if checkpoint_path is not None:
            state_dict_loaded = torch.load(checkpoint_path, map_location="cpu")[
                "state_dict"
            ]
            for k, v in state_dict_loaded.items():
                state_dict[k.replace("model.", "")] = v
        self.model.load_state_dict(state_dict, strict=True)
        self.loss_train = torch.nn.CrossEntropyLoss(
            ignore_index=self.ignore_index, reduction="mean", label_smoothing=0.0
        )
        # self.loss_train = FlashCELoss(ignore_index=self.ignore_index, reduction='mean', label_smoothing=0.0,
        #                              inplace_backward=True)

        self.intern_log = []
        self.log_lists = defaultdict(list)

        self.validation_step_outputs = defaultdict(list)
        # build choices dict
        self.choices_dict = {}
        self.choices_dict["n_layer_choices"] = cfg_model.layer_choices
        self.choices_dict["n_head_choices"] = cfg_model.head_choices
        self.choices_dict["embed_dim_choices"] = cfg_model.embed_choices
        self.choices_dict["mlp_ratio_choices"] = cfg_model.mlp_ratio_choices
        self.choices_dict["bias_choices"] = cfg_model.bias_choices
        self.scheme = cfg_model.sampling_scheme
        self.train_strategy = cfg_model.train_strategy
        self.sandwhich_random = cfg_model.sandwhich_random
        self.init_max_min()
        self.arch_sampled = arch_sampled
        # self.automatic_optimization = False

    def init_max_min(self):
        self.config_max = sample_config_max(
            self.choices_dict, layer_sampling_scheme=self.scheme
        )
        self.config_min = sample_config_min(
            self.choices_dict, layer_sampling_scheme=self.scheme
        )
        self.config_mid = sample_config_mid(
            self.choices_dict, layer_sampling_scheme=self.scheme
        )

    def get_arch_sampled(self):

        return self.arch_sampled

    def training_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0):

        sampled_config = self.get_arch_sampled()
        sample_intermediate_size = [
            sampled_config["sample_mlp_ratio"][i] * sampled_config["sample_embed_dim"]
            for i in range(len(sampled_config["sample_mlp_ratio"]))
        ]
        self.model.set_sample_config(
            sampled_config["sample_embed_dim"],
            sample_intermediate_size,
            sampled_config["sample_n_head"],
            sampled_config["sample_n_layer"],
            sampled_config["sample_bias"],
            sampled_config["sample_layer_indices"],
        )
        logits = self.model(idx=batch["src_seq"])
        labels = batch["trg_seq"].view(-1)
        if (
            self.cfg_model.use_inplace_kd
            and self.local_rank != 0
            and self.global_step > 5000
        ):
            with torch.no_grad():
                sampled_config = self.config_max
                sample_intermediate_size = [
                    sampled_config["sample_mlp_ratio"][i]
                    * sampled_config["sample_embed_dim"]
                    for i in range(len(sampled_config["sample_mlp_ratio"]))
                ]
                self.model.set_sample_config(
                    sampled_config["sample_embed_dim"],
                    sample_intermediate_size,
                    sampled_config["sample_n_head"],
                    sampled_config["sample_n_layer"],
                    sampled_config["sample_bias"],
                    sampled_config["sample_layer_indices"],
                )
                # self.model.set_sample_config(sampled_config["sample_embed_dim"], sampled_config["sample_mlp_ratio"]*sampled_config["sample_embed_dim"], sampled_config["sample_n_head"], sampled_config["sample_n_layer"], sampled_config["sample_bias"], sampled_config["sample_layer_indices"])
                logits_teacher = self.model(idx=batch["src_seq"])
                logits_teacher = logits_teacher.detach().view(
                    -1, logits_teacher.size(-1)
                )
        else:
            logits_teacher = labels
        loss = self.loss_train(logits.view(-1, logits.size(-1)), logits_teacher)
        # loss = loss#/n
        loss_value = loss.detach()  # *n
        self.log(
            "train/loss",
            loss_value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": loss}

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0):

        sampled_config = self.arch_sampled
        sample_intermediate_size = [
            sampled_config["sample_mlp_ratio"][i] * sampled_config["sample_embed_dim"]
            for i in range(len(sampled_config["sample_mlp_ratio"]))
        ]
        self.model.set_sample_config(
            sampled_config["sample_embed_dim"],
            sample_intermediate_size,
            sampled_config["sample_n_head"],
            sampled_config["sample_n_layer"],
            sampled_config["sample_bias"],
            sampled_config["sample_layer_indices"],
        )
        with torch.no_grad():
            logits = self.model(idx=batch["src_seq"]).detach()

            loss = self.loss_train(
                logits.view(-1, logits.size(-1)), batch["trg_seq"].view(-1)
            )

        if dataloader_idx == 0:
            self.log(
                "val/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return_dict = {
            "loss": loss,
            "batch_size": torch.FloatTensor([batch["trg_len"].shape[0]]),
            "batch_length": torch.mean(batch["trg_len"].detach().float()),
            "num_loss_tokens": torch.sum(batch["trg_len"]),
        }

        count = torch.sum(batch["trg_len"], dtype=torch.float)
        log_probs = loss * count
        preds = logits.argmax(dim=-1).view(-1)
        target = batch["trg_seq"].view(-1)
        idx = target != self.ignore_index
        accuracy = torch.sum(preds[idx] == target[idx])

        return_dict.update(
            {"accuracy": accuracy, "log_probs": log_probs, "count": count}
        )

        self.validation_step_outputs[dataloader_idx].append(return_dict)

        return return_dict

    def on_validation_epoch_end(self):
        values = ["log_probs", "accuracy", "count"]

        assert len(self.val_sets_name) == len(self.validation_step_outputs)

        for dataset_idx, dataset_name in enumerate(self.val_sets_name):

            output = self.validation_step_outputs[dataset_idx]
            summed_values = {k: 0 for k in values}
            for out_dict in output:
                for key in values:
                    summed_values[key] += out_dict[key]

            ppl = torch.exp(summed_values["log_probs"] / summed_values["count"])
            accuracy = summed_values["accuracy"] / summed_values["count"]
            metrics = {"ppl": ppl, "acc": accuracy}
            # print(metrics)
            for name, value in metrics.items():
                self.log(
                    f"val/{dataset_name}/{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

        self.current_metrics = self.all_reduce(self.all_gather(metrics))
        # else:
        #    self.current_metrics = None
        self.validation_step_outputs.clear()

    def all_reduce(self, metrics: Dict):
        metrics_reduce = {}
        for key, value in metrics.items():
            mean_value = value.mean()
            metrics_reduce[key] = mean_value.item()

        return metrics_reduce

    def configure_optimizers(self):

        if (
            "optimizer_param_grouping" in self.cfg_train
        ):  # Set zero weight decay for some params
            parameters = group_parameters_for_optimizer(
                self.model,
                self.cfg_train.optimizer,
                **self.cfg_train.optimizer_param_grouping,
            )
        else:
            parameters = self.model.parameters()

        optimizer = instantiate(self.cfg_train.optimizer, parameters)

        # Log optimizer info
        for i, g in enumerate(optimizer.param_groups):
            ntensors = len(g["params"])
            nparams = sum(p.numel() for p in g["params"])
            hparams = {k: v for k, v in g.items() if k != "params"}
            self.py_logger.info(
                f"Optimizer group {i}: {ntensors} tensors, {nparams} parameters, {hparams}"
            )

        if "scheduler" not in self.cfg_train:
            return optimizer
        else:
            lr_lambda = get_learning_rate_schedule(self.cfg_train.scheduler)

            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda, last_epoch=-1
            )

            return [optimizer], {
                "scheduler": lr_scheduler,
                "interval": self.cfg_train.get("scheduler_interval", "step"),
                "monitor": self.cfg_train.get("scheduler_monitor", "val/loss"),
            }

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: torch.optim.Optimizer
    ):
        # https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html#set-grads-to-none
        # TD [2022-04-30]: DeepSpeed optimizer uses the kwarg set_grad_to_none instead of set_to_none
        if "set_to_none" in inspect.signature(optimizer.zero_grad).parameters:
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad()

    # def on_train_epoch_start(self):
    #    random.seed(self.current_epoch)
