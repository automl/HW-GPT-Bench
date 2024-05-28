import os
import sys
import socket
import argparse
import collections
import yaml

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import random
import logging
import torch.cuda
import pytorch_lightning as pl
import numpy as np
from typing import List, Dict, Any
from data_collection.pl_gpt.data.lm_datamodule import PlArrowFileModule
from data_collection.pl_gpt.pl_module.lm_trainer_configurable import (
    LanguageModelTrainer,
)

from data_collection.pl_gpt.utils.configuration import Config
from data_collection.pl_gpt.utils.instantiate import instantiate
from data_collection.pl_gpt.utils.folder_manager import get_experiment_folder

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["HF_DATASETS_CACHE"] = "/p/scratch/ccstdl/sukthanker1/datasets/cache"
os.environ["HF_HOME"] = "/p/scratch/ccstdl/sukthanker1/model/cache"


def bold(msg:str):
    return f"\033[1m{msg}\033[0m"


def main(cfg:Any):
    """
    Launch pretraining
    """

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

    # if do_resume_training:
    #     model_module = LanguageModelTrainer.load_from_checkpoint(exp_folder / "last.ckpt" / 'checkpoint' )
    #     # version = f"version_{cfg.resume_training.split('version_')[1].split('/')[0]}"
    # else:
    model_module = LanguageModelTrainer(
        cfg_train=cfg.train,
        cfg_model=cfg.model,
        py_logger=logger,
        val_sets_name=data_module.val_sets_name,
        ignore_index=data_module.ignore_index,
    )

    if is_rank_zero:

        def count_parameters(parameters:int):
            return sum(p.numel() for p in parameters if p.requires_grad)

        logger.info(
            f"#### trainable_parameters {count_parameters(model_module.parameters())}"
        )

        def print_model_param_stats(model:torch.nn.Module):
            for idx, (name, params) in enumerate(model.named_parameters()):
                logger.info(
                    f"{idx:03d} {name:70} shape:{str(list(params.shape)):12} mean:{params.mean():8.4f} std:{params.std():8.6f} grad: {params.requires_grad}"
                )

        print_model_param_stats(model_module.model)

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
    if "callbacks" in config:
        for cb_name, cb_conf in config.callbacks.items():
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

    trainer.fit(
        model=model_module,
        datamodule=data_module,
        ckpt_path=exp_folder / "last.ckpt" if do_resume_training else None,
    )

    logger.info(f"Finished saving model weights on rank {rank}")
    logger.info("End training!")


if __name__ == "__main__":

    from functools import reduce  # forward compatibility for Python 3
    import operator

    def update(d:Dict, u:Dict):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def getFromDict(dataDict:Dict, mapList:List):
        return reduce(operator.getitem, mapList, dataDict)

    def setInDict(dataDict:Dict, mapList:List, value:Any):
        getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

    def convert_string_value(value:Any):
        if value in ("false", "False"):
            value = False
        elif value in ("true", "True"):
            value = True
        else:
            try:
                value = int(value)
            except Exception:
                try:
                    value = float(value)
                except Exception:
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

    config = Config(config_dict=config_dict)

    main(cfg=config)
