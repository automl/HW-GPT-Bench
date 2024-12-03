import socket
import sys


import logging
from argparse import ArgumentParser
from pathlib import Path

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.baselines import (
    RandomSearch,
    GridSearch,
    MOREA,
    NSGA2,
    MORandomScalarizationBayesOpt,
    MOASHA,
    EHVI,
)
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import choice
from baselines.local_search import LS
from syne_tune.optimizer.schedulers.multiobjective.linear_scalarizer import (
    LinearScalarizedScheduler,
)
import os
import pickle
from lib.utils import search_spaces

# Configuration space (or search space)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    # socket.create_connection(("www.google.com", 10))
    parser = ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        choices=(
            "RS",
            "BO",
            "Grid",
            "MOREA",
            "LS",
            "NSGA2",
            "LSBO",
            "RSBO",
            "MOASHA",
            "EHVI",
        ),
        default="RS",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=31415927,
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_wallclock_time",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default="mogpt",
    )
    parser.add_argument(
        "--max_num_layers",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--device_1",
        type=str,
        default="rtx2080",
    )
    parser.add_argument(
        "--device_2",
        type=str,
        default="a6000",
    )
    parser.add_argument("--objective_1", type=str, default="latencies")
    parser.add_argument("--objective_2", type=str, default="energies")
    parser.add_argument("--search_space", type=str, default="s")
    parser.add_argument("--surrogate_type", type=str, default="conformal_quantile")
    parser.add_argument("--type", type=str, default="quantile")
    args, _ = parser.parse_known_args()
    search_space = search_spaces[args.search_space]
    max_layers = max(search_space["n_layer_choices"])
    config_space = {
        "search_space": args.search_space,
        "device_1": args.device_1,
        "device_2": args.device_2,
        "objective_1": args.objective_1,
        "objective_2": args.objective_2,
        "surrogate_type": args.surrogate_type,
        "num_layers": choice(search_space["n_layer_choices"]),
        "embed_dim": choice(search_space["embed_dim_choices"]),
        "bias": choice([True, False]),
    }
    for i in range(max_layers):
        config_space[f"mlp_ratio_{i}"] = choice(search_space["mlp_ratio_choices"])
        config_space[f"num_heads_{i}"] = choice(search_space["n_head_choices"])

    # Here, we specify the training script we want to tune
    # - `mode` and `metric` must match what is reported in the training script
    # - Metrics need to be reported after each epoch, `resource_attr` must match
    #   what is reported in the training script
    train_file = "gpt_objective_3d.py"
    entry_point = Path(__file__).parent / train_file
    max_resource_level = 1  # Maximum number of training epochs
    mode = "min"
    metrics = ["perplexity", "hw_metric_1", "hw_metric_2"]
    resource_attr = "epoch"
    max_resource_attr = "epochs"

    # Additional fixed parameters  [2]
    config_space.update(
        {
            max_resource_attr: max_resource_level,
            "dataset_path": "./",
        }
    )

    # Local backend: Responsible for scheduling trials  [3]
    # The local backend runs trials as sub-processes on a single instance
    trial_backend = LocalBackend(entry_point=str(entry_point))

    # Scheduler: Depends on `args.method`  [4]
    scheduler = None
    # Common scheduler kwargs
    method_kwargs_single = dict(
        metric=metrics[0],
        mode=mode,
        random_seed=args.random_seed,
        max_resource_attr=max_resource_attr,
        search_options={"num_init_random": 8},
    )
    method_kwargs_multi = dict(
        metric=metrics,
        mode=["min", "min", "min"],
        random_seed=args.random_seed,
        max_resource_attr=max_resource_attr,
        search_options={"num_init_random": 8},
    )
    method_kwargs_moasha = dict(metrics=metrics, mode=["min", "min", "min"])
    sch_type = "promotion" if args.method.endswith("PROM") else "stopping"
    if args.method == "RS":
        scheduler = RandomSearch(config_space, **method_kwargs_single)
    elif args.method == "Grid":
        print(method_kwargs_single)
        scheduler = GridSearch(config_space, **method_kwargs_single)
    elif args.method == "MOREA":
        print(method_kwargs_multi)
        scheduler = MOREA(config_space, **method_kwargs_multi)
    elif args.method == "LS":
        scheduler = LS(config_space, **method_kwargs_multi)
    elif args.method == "NSGA2":
        scheduler = NSGA2(config_space, **method_kwargs_multi)
    elif args.method == "LSBO":
        scheduler = LinearScalarizedScheduler(
            config_space, searcher="bayesopt", **method_kwargs_multi
        )
    elif args.method == "RSBO":
        scheduler = MORandomScalarizationBayesOpt(config_space, **method_kwargs_multi)
    elif args.method == "MOASHA":
        scheduler = MOASHA(
            config_space,
            time_attr="st_worker_time",
            grace_period=1,
            max_t=5,
            reduction_factor=3,
            **method_kwargs_moasha,
        )
    elif args.method == "EHVI":
        scheduler = EHVI(config_space, **method_kwargs_multi)
    else:
        raise NotImplementedError(args.method)

    # Stopping criterion: We stop after `args.max_wallclock_time` seconds
    # [5]
    stop_criterion = StoppingCriterion(max_wallclock_time=args.max_wallclock_time)

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=args.n_workers,
        tuner_name=args.experiment_tag,
        metadata={
            "seed": args.random_seed,
            "algorithm": args.method,
            "tag": args.experiment_tag,
        },
    )

    tuner.run()
    from syne_tune.experiments import load_experiment

    print(tuner.name)
    df = load_experiment(tuner.name).results
    configs = []
    runtime_traj = []
    energy = []
    latency = []
    perplexity = []
    print(df.head())
    for trial, trial_df in df.groupby("trial_id"):
        idx = trial_df["perplexity"].idxmin()
        runtime_traj.append(float(trial_df.st_tuner_time.iloc[-1]))
        perplexity.append(trial_df["perplexity"].values)
        energy.append(trial_df["hw_metric_1"].values)
        latency.append(trial_df["hw_metric_2"].values)
        config = {}
        for hyper in config_space.keys():
            c = trial_df.iloc[0]["config_" + hyper]
            config[hyper] = c
        configs.append(config)
        print(configs)
    results = {
        "configs": configs,
        "runtime_traj": runtime_traj,
        "perplexity": perplexity,
        "hw_metric_1": energy,
        "hw_metric_2": latency,
    }
    search_space_path = "results_gpt_baselines_3d_" + str(args.search_space)
    os.makedirs(search_space_path, exist_ok=True)
    method_path = search_space_path + "/" + args.method + "/"
    os.makedirs(method_path, exist_ok=True)
    objectiv_path = method_path + args.objective_1 + "_" + args.objective_2 + "/"
    os.makedirs(objectiv_path, exist_ok=True)
    save_path = (
        objectiv_path
        + args.experiment_tag
        + "_"
        + args.device_1
        + "_"
        + args.device_2
        + "_"
        + str(args.random_seed)
        + ".pickle"
    )
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
