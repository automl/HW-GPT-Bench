from collections import defaultdict
import pytorch_lightning as pl
import torch
from hwgpt.model.gpt.model import GPT
from hwgpt.model.gpt.utils import sample_config
from data_collection.pl_gpt.utils.metriclogger import MetricLogger
import torch.distributed as dist
from typing import List


class LanguageModelEvaluator(pl.LightningModule):
    """
    PTL wrapper class for model training
    """

    def __init__(
        self,
        cfg_train,
        cfg_model,
        checkpoint_path,
        py_logger,
        val_sets_name,
        ignore_index,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.cfg_train = cfg_train
        self.cfg_model = cfg_model

        self.val_sets_name = val_sets_name
        self.ignore_index = ignore_index
        self.py_logger = py_logger

        self.model = GPT(cfg_model)
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

    def set_sample_config(self, model_config):
        self.model_config = model_config

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        sampled_config = self.model_config
        # print(sampled_config)
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
                f"val/loss",
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
            # g

        # check rank is 0
        # if self.global_rank == 0:
        # if self.local_rank == 0:
        self.current_metrics = self.all_reduce(self.all_gather(metrics))
        # else:
        #    self.current_metrics = None
        self.validation_step_outputs.clear()

    def all_reduce(self, metrics):
        metrics_reduce = {}
        for key, value in metrics.items():
            mean_value = value.mean()
            metrics_reduce[key] = mean_value.item()

        return metrics_reduce


# test
def bold(msg):
    return f"\033[1m{msg}\033[0m"


if __name__ == "__main__":
    from functools import reduce  # forward compatibility for Python 3
    import operator
    import os, sys, socket
    import argparse, collections, yaml

    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    import random
    import logging
    import torch.cuda
    import pytorch_lightning as pl
    import numpy as np

    from pl_gpt.data.lm_datamodule_nas import PlArrowFileModule
    from pl_gpt.pl_module.lm_trainer_configurable import LanguageModelTrainer

    from pl_gpt.utils.configuration import Config
    from pl_gpt.utils.instantiate import instantiate
    from pl_gpt.utils.folder_manager import get_experiment_folder

    def update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def getFromDict(dataDict, mapList):
        return reduce(operator.getitem, mapList, dataDict)

    def setInDict(dataDict, mapList, value):
        getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

    def convert_string_value(value):
        if value in ("false", "False"):
            value = False
        elif value in ("true", "True"):
            value = True
        else:
            try:
                value = int(value)
            except:
                try:
                    value = float(value)
                except:
                    pass
        return value

    print("CUDA AVAILABLE", torch.cuda.is_available())
    print("CUDA DEVICES", torch.cuda.device_count())

    if socket.gethostname() == "tower":
        default_config_name = "default_config.yaml"
    else:
        default_config_name = "default_config.yaml"

    parser = argparse.ArgumentParser(description="Train GPT2 LLM")
    parser.add_argument(
        "-c", "--config", type=str, default=default_config_name, help="config file name"
    )

    # parser.add_argument('args', nargs=argparse.REMAINDER)

    args, unknown_args = parser.parse_known_args()

    config_name = args.config
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"

    config_file = os.path.join("config", args.config)
    with open(config_file, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)

    for arg in unknown_args:
        if "=" in arg:
            keys = arg.split("=")[0].split(".")
            value = convert_string_value(arg.split("=")[1])
            print(keys, value)
            setInDict(config_dict, keys, value)
        else:
            raise UserWarning(f"argument unknown: {arg}")

    cfg = Config(config_dict=config_dict)

    import os, sys, socket
    import argparse, collections, yaml

    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    import random
    import logging
    import torch.cuda
    import pytorch_lightning as pl
    import numpy as np

    from pl_gpt.data.lm_datamodule_nas import PlArrowFileModule

    from pl_gpt.utils.configuration import Config
    from pl_gpt.utils.instantiate import instantiate
    from pl_gpt.utils.folder_manager import get_experiment_folder

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import os

    torch.set_float32_matmul_precision("medium")

    if os.environ.get("LOCAL_RANK") is None or os.environ.get("LOCAL_RANK") == 0:
        is_rank_zero = True
        rank = 0
    else:
        is_rank_zero = False
        rank = os.environ.get("LOCAL_RANK")

    seed = int(cfg.train.seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cfg.resume_training:
        exp_folder = get_experiment_folder(
            **cfg.experiment, new_folder=False, count_folder=False
        )
        # if file last.ckpt exisits, resume training
        if (exp_folder / "last.ckpt").exists():
            do_resume_training = True
        else:
            do_resume_training = False

    else:
        do_resume_training = False
        exp_folder = get_experiment_folder(**cfg.experiment, new_folder=is_rank_zero)

    logger = logging.getLogger(__name__)

    if is_rank_zero:
        cfg.save_config(exp_folder)

        logging.basicConfig(
            format="[%(asctime)s][%(levelname)s][%(name)s] - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(exp_folder / "logfile.txt"),
            ],
        )

        logger.info(bold("######################################################"))
        logger.info(bold("########          START   TRAINING          ##########"))
        logger.info(bold("######################################################"))

        logger.info(f"########  Project:    {cfg.experiment.project_name}")
        logger.info(f"########  Session:    {cfg.experiment.session_name}")
        logger.info(f"########  Experiment: {cfg.experiment.experiment_name}")
        logger.info(f"save logs and checkpoints in: {exp_folder.as_posix()}")

        logger.info(bold("############### CONFIGURATION"))
        logger.info("Data args")
        logger.info(cfg.lm_data)
        logger.info("Trainer args")
        logger.info(cfg.trainer)
        logger.info("Train args")
        logger.info(cfg.train)
        logger.info("Callbacks args")
        logger.info(cfg.callbacks)
        logger.info("Deepspeed args")
        logger.info(cfg.deepspeed)
        logger.info("Optimizer args")
        logger.info(cfg.train.optimizer)
        logger.info("Model args")
        logger.info(cfg.model)

    # If dataloader is already multiprocessing, skip this
    # if cfg.trainer.devices > 1 and cfg.task == "lm":
    #     os.environ["TOKENIZERS_PARALLELISM"] = "False"

    # Set seed before initializing model

    # set up huggingface-style model config
    # uses custom config class for extra args

    logger.info(bold(f"############### LOAD DATA on rank {rank}"))

    cfg_lm_data = {**cfg.lm_data}
    cfg_lm_data["num_gpu_worker"] = cfg.trainer.devices * cfg.trainer.num_nodes
    data_module = PlArrowFileModule(**cfg_lm_data)
    # data_module = InMemoryDataModule(**cfg.data)

    assert (cfg.lm_data.max_sample_len) % 128 == 0
    print(data_module.seq_vocab_size)
    cfg.model.vocab_size = data_module.seq_vocab_size
    cfg.model.padded_vocab_size = data_module.trg_vocab_size
    cfg.model.max_len = cfg.lm_data.max_sample_len
    logger.info(f"#### vocab size: {data_module.seq_vocab_size}")
    trainer_pl = LanguageModelEvaluator(
        cfg_train=cfg.train,
        cfg_model=cfg.model,
        checkpoint_path=cfg.model.checkpoint_path,
        py_logger=logger,
        val_sets_name=data_module.val_sets_name,
        ignore_index=data_module.ignore_index,
    )
    sampled_config = sample_config(
        trainer_pl.choices_dict, layer_sampling_scheme=trainer_pl.scheme, seed=0
    )

    if is_rank_zero:

        def count_parameters(parameters):
            return sum(p.numel() for p in parameters if p.requires_grad)

        logger.info(
            f"#### trainable_parameters {count_parameters(trainer_pl.parameters())}"
        )

        def print_model_param_stats(model):
            for idx, (name, params) in enumerate(model.named_parameters()):
                logger.info(
                    f"{idx:03d} {name:70} shape:{str(list(params.shape)):12} mean:{params.mean():8.4f} std:{params.std():8.6f} grad: {params.requires_grad}"
                )

        print_model_param_stats(trainer_pl.model)

    if do_resume_training:
        logger.info(bold(f"############### RESUME TRAINING on rank {rank}"))

    logger.info(f"#### Load logger on rank {rank}")
    # Init lightning loggers
    # training_logger: List[LightningLoggerBase] = []
    # if "logger" in cfg:
    #     for lg_name, lg_conf in cfg.logger.items():
    #         if lg_conf is not None and "_target_" in lg_conf:
    #             logger.info(f"Instantiating logger <{lg_name}>")
    #             training_logger.append(instantiate(lg_conf, save_dir=exp_folder))

    training_logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir=exp_folder,
        name="",
        version="tb",
        #    log_graph: False
        #    default_hp_metric: True
        prefix="",
    )

    logger.info(f"#### Load callbacks on rank {rank}")
    # Init lightning callbacks
    callbacks: List[pl.Callback] = []
    if "callbacks" in cfg:
        for cb_name, cb_conf in cfg.callbacks.items():
            if cb_conf is not None and "_target_" in cb_conf:
                logger.info(f"Instantiating callback <{cb_name}>")
                if "dirpath" in cb_conf:
                    cb_conf["dirpath"] = exp_folder
                callbacks.append(instantiate(cb_conf))

    logger.info(f"#### Load strategy on rank {rank}")

    if cfg.trainer.devices == 1:
        strategy = "ddp"

        strategy = pl.strategies.DDPStrategy(
            find_unused_parameters=True, static_graph=False
        )
        # strategy = pl.strategies.DeepSpeedStrategy(
        #     **cfg.deepspeed,
        #     remote_device=None,  # Initialize directly on GPUs instead of CPU (ZeRO-3)
        # )

    else:
        strategy = pl.strategies.DDPStrategy(
            find_unused_parameters=True, static_graph=False
        )

        # strategy = pl.strategies.DeepSpeedStrategy(
        #     **cfg.deepspeed,
        #     remote_device=None,  # Initialize directly on GPUs instead of CPU (ZeRO-3)
        # )

    plugins = []
    # plugins = [SLURMEnvironment(auto_requeue=False)]

    logger.info(bold(f"############### TRAINER on rank {rank}"))

    trainer = instantiate(
        cfg.trainer,
        instance=pl.Trainer,
        callbacks=callbacks,
        plugins=plugins,
        strategy=strategy,
        logger=training_logger,
    )

    logger.info(f"Starting training on rank {rank}")
    data_module.prepare_data()
    data_module.setup(stage="fit")
    for i in range(1000):
        sampled_config = sample_config(
            trainer_pl.choices_dict, layer_sampling_scheme=trainer_pl.scheme, seed=i
        )
        trainer_pl.set_sample_config(sampled_config)
        trainer.validate(trainer_pl, data_module.val_nas_dataloader())
        # print agregated stats across all processes only once
        if is_rank_zero:
            print(trainer_pl.current_metrics)
