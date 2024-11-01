"""
Examples of code for visualizations.
"""

import numpy as np
import uncertainty_toolbox as uct
import pickle
from plotting.utils import get_calibration_metrics

from plotting.utils import plot_xy
import matplotlib.pyplot as plt


def cpu_map(deivce):
    if device == "cpu_xeon_silver":
        return "Xeon Silver"
    elif device == "cpu_xeon_gold":
        return "Xeon Gold"
    elif device == "cpu_amd_7502":
        return "AMD 7502"
    elif device == "cpu_amd_7513":
        return "AMD 7513"
    elif device == "cpu_amd_7452":
        return "AMD 7452"
    else:
        return device


def make_confidence_plots(args):
    if args.search_space == "s":
        mul_factor = 0.5
    elif args.search_space == "m":
        mul_factor = 1
    else:
        mul_factor = 4
    if "cpu" in args.device:
        device = cpu_map(args.device) + " "
    else:
        device = args.device.upper() + " "
    if args.objective == "latencies":
        device += "(Lat)"
    else:
        device += "(En)"
    device = device + " GPT-" + args.search_space.upper()
    uct.viz.set_style()
    uct.viz.update_rc("text.usetex", True)  # Set to True for system latex
    uct.viz.update_rc("font.size", 14)  # Set font size
    uct.viz.update_rc("xtick.labelsize", 14)  # Set font size for xaxis tick labels
    uct.viz.update_rc("ytick.labelsize", 14)  # Set font size for yaxis tick labels
    plt.rcParams["figure.figsize"] = (5, 5)
    # Set random seed
    a_autogluon, a_xgb, a_mix, a_lightgbm = get_calibration_metrics(args)
    np.random.seed(11)
    pred_mean_list = [np.array(a_autogluon[1]["Target_Avg"])]

    pred_std_list = [np.array(a_autogluon[1]["Target_Std"])]
    # Generate synthetic predictive uncertainty results
    n_obs = 650
    x = (
        np.sum(
            np.array(a_autogluon[0].drop(columns=["Target_Avg", "Target_Std"])), axis=-1
        )
        / 1000
    )
    y = np.array(a_autogluon[0]["Target_Avg"])
    # Save figure (set to True to save)
    savefig = True
    # Loop through, make plots, and compute metrics
    idx_counter = 0
    for i, pred_mean in enumerate(pred_mean_list):
        for j, pred_std in enumerate(pred_std_list):
            mace = uct.mean_absolute_calibration_error(pred_mean, pred_std, y)
            rmsce = uct.root_mean_squared_calibration_error(pred_mean, pred_std, y)
            ma = uct.miscalibration_area(pred_mean, pred_std, y)
            y = y * mul_factor
            pred_mean = pred_mean * mul_factor
            pred_std = pred_std * mul_factor
            if args.objective == "latencies":
                ylims = [min(y / 1000) - 0.5, max(y / 1000) + 0.5]
            else:
                ylims = [min(y) - 0.5, max(y) + 0.5]
            idx_counter += 1

            if args.objective == "latencies":
                plot_xy(
                    pred_mean / 1000,
                    pred_std / 1000,
                    y / 1000,
                    x,
                    n_subset=300,
                    ylims=ylims,
                    xlims=[min(x), max(x)],
                    ax=plt.gca(),
                    color="blue",
                    plot_true=True,
                    device=device,
                    metric="Latency (ms)",
                )
            else:
                plot_xy(
                    pred_mean,
                    pred_std,
                    y,
                    x,
                    n_subset=300,
                    ylims=ylims,
                    xlims=[min(x), max(x)],
                    ax=plt.gca(),
                    color="blue",
                    plot_true=True,
                    device=device,
                    metric="Energy (Wh)",
                )
            # plot_intervals_ordered(
            # pred_mean, pred_std, y, n_subset=100, ylims=ylims,color="blue",ax=plt.gca(),label="AutoGluon",device=device)
            # uct.plot_calibration(pred_mean, pred_std, y,ax=plt.gca(),color="blue",curve_label="AutoGluon",device="V100 (En)")

            print(f"MACE: {mace}, RMSCE: {rmsce}, MA: {ma}")
    # List of predictive means and standard deviations
    pred_mean_list = [np.array(a_xgb[1])]

    pred_std_list = [np.array(a_xgb[2])]
    # Generate synthetic predictive uncertainty results
    n_obs = 650
    x = (
        np.sum(np.array(a_xgb[0].drop(columns=["Target_Avg", "Target_Std"])), axis=-1)
        / 1000
    )
    y = np.array(a_xgb[0]["Target_Avg"])
    # Save figure (set to True to save)
    savefig = True
    # Loop through, make plots, and compute metrics
    idx_counter = 0
    for i, pred_mean in enumerate(pred_mean_list):
        for j, pred_std in enumerate(pred_std_list):
            mace = uct.mean_absolute_calibration_error(pred_mean, pred_std, y)
            rmsce = uct.root_mean_squared_calibration_error(pred_mean, pred_std, y)
            ma = uct.miscalibration_area(pred_mean, pred_std, y)

            idx_counter += 1
            y = y * mul_factor
            pred_mean = pred_mean * mul_factor
            pred_std = pred_std * mul_factor
            if args.objective == "latencies":
                ylims = [min(y / 1000) - 0.5, max(y / 1000) + 0.5]
            else:
                ylims = [min(y) - 0.5, max(y) + 0.5]
            idx_counter += 1
            if args.objective == "latencies":
                plot_xy(
                    pred_mean / 1000,
                    pred_std / 1000,
                    y / 1000,
                    x,
                    n_subset=300,
                    ylims=ylims,
                    xlims=[min(x), max(x)],
                    ax=plt.gca(),
                    color="green",
                    plot_true=False,
                    method="Ensemble (XGB)",
                    metric="Latency (ms)",
                    device=device,
                )
            else:
                plot_xy(
                    pred_mean,
                    pred_std,
                    y,
                    x,
                    n_subset=300,
                    ylims=ylims,
                    xlims=[min(x), max(x)],
                    ax=plt.gca(),
                    color="green",
                    plot_true=False,
                    method="Ensemble (XGB)",
                    device=device,
                    metric="Energy (Wh)",
                )
            # plot_intervals_ordered(
            # pred_mean, pred_std, y, n_subset=100, ylims=ylims,color="green",ax=plt.gca(),label="Ensemble(XGB)",device=device)
            # uct.plot_calibration(pred_mean, pred_std, y, ax=plt.gca(),color="green",curve_label="Ensemble_XGB",ideal_flag=ideal_flag,device="V100 (En)")

            print(f"MACE: {mace}, RMSCE: {rmsce}, MA: {ma}")
    pred_mean_list = [np.array(a_mix[1])]
    pred_std_list = [np.array(a_mix[2])]
    # Generate synthetic predictive uncertainty results
    x = (
        np.sum(np.array(a_mix[0].drop(columns=["Target_Avg", "Target_Std"])), axis=-1)
        / 1000
    )
    y = np.array(a_mix[0]["Target_Avg"])
    # Save figure (set to True to save)
    # Loop through, make plots, and compute metrics
    idx_counter = 0
    for i, pred_mean in enumerate(pred_mean_list):
        for j, pred_std in enumerate(pred_std_list):
            mace = uct.mean_absolute_calibration_error(pred_mean, pred_std, y)
            rmsce = uct.root_mean_squared_calibration_error(pred_mean, pred_std, y)
            ma = uct.miscalibration_area(
                pred_mean,
                pred_std,
                y,
            )
            y = y * mul_factor
            pred_mean = pred_mean * mul_factor
            pred_std = pred_std * mul_factor
            idx_counter += 1
            if args.objective == "latencies":
                ylims = [min(y / 1000) - 0.5, max(y / 1000) + 0.5]
            else:
                ylims = [min(y) - 0.5, max(y) + 0.5]

            if args.objective == "latencies":
                plot_xy(
                    pred_mean / 1000,
                    pred_std / 1000,
                    y / 1000,
                    x,
                    n_subset=300,
                    ylims=ylims,
                    xlims=[min(x), max(x)],
                    ax=plt.gca(),
                    color="red",
                    plot_true=False,
                    method="Ensemble (MIX)",
                    metric="Latency (ms)",
                    device=device,
                )
            else:
                plot_xy(
                    pred_mean,
                    pred_std,
                    y,
                    x,
                    n_subset=300,
                    ylims=ylims,
                    xlims=[min(x), max(x)],
                    ax=plt.gca(),
                    color="red",
                    plot_true=False,
                    method="Ensemble (MIX)",
                    device=device,
                    metric="Energy (Wh)",
                )
            # plot_intervals_ordered(
            # pred_mean, pred_std, y, n_subset=100, ylims=ylims,color="red",ax=plt.gca(),label="Ensemble(MIX)",device=device)
            # uct.plot_calibration(pred_mean, pred_std, y,ax=plt.gca(),color="red",curve_label="Ensemble_MIX",ideal_flag=ideal_flag,device="V100 (En)")

            print(f"MACE: {mace}, RMSCE: {rmsce}, MA: {ma}")
    pred_mean_list = [np.array(a_lightgbm[1])]

    pred_std_list = [np.array(a_lightgbm[2])]
    # Generate synthetic predictive uncertainty results
    x = (
        np.sum(
            np.array(a_lightgbm[0].drop(columns=["Target_Avg", "Target_Std"])), axis=-1
        )
        / 1000
    )
    y = np.array(a_lightgbm[0]["Target_Avg"])
    # Save figure (set to True to save)
    # Loop through, make plots, and compute metrics
    idx_counter = 0
    for i, pred_mean in enumerate(pred_mean_list):
        for j, pred_std in enumerate(pred_std_list):
            mace = uct.mean_absolute_calibration_error(pred_mean, pred_std, y)
            rmsce = uct.root_mean_squared_calibration_error(pred_mean, pred_std, y)
            ma = uct.miscalibration_area(pred_mean, pred_std, y)
            y = y * mul_factor
            pred_mean = pred_mean * mul_factor
            pred_std = pred_std * mul_factor
            if args.objective == "latencies":
                ylims = [min(y / 1000) - 0.5, max(y / 1000) + 0.5]
            else:
                ylims = [min(y) - 0.5, max(y) + 0.5]
            idx_counter += 1

            if args.objective == "latencies":
                plot_xy(
                    pred_mean / 1000,
                    pred_std / 1000,
                    y / 1000,
                    x,
                    n_subset=300,
                    ylims=ylims,
                    xlims=[min(x), max(x)],
                    ax=plt.gca(),
                    color="orange",
                    plot_true=False,
                    method="Ensemble (LightGBM)",
                    metric="Latency (ms)",
                    device=device,
                )
            else:
                plot_xy(
                    pred_mean,
                    pred_std,
                    y,
                    x,
                    n_subset=300,
                    ylims=ylims,
                    xlims=[min(x), max(x)],
                    ax=plt.gca(),
                    color="orange",
                    plot_true=False,
                    method="Ensemble(LightGBM)",
                    device=device,
                    metric="Energy (Wh)",
                )
            # plot_intervals_ordered(
            # pred_mean, pred_std, y, n_subset=100, ylims=ylims,color="orange",true_flag=True,ax=plt.gca(),label="Ensemble(LightGBM)", device=device)
            # uct.plot_calibration(pred_mean, pred_std, y,ax=plt.gca(),color="orange",curve_label="Ensemble_LightGBM",ideal_flag=ideal_flag,device="V100 (En)")

            print(f"MACE: {mace}, RMSCE: {rmsce}, MA: {ma}")
    # plt.plot(x, y/1000, "--", linewidth=3.0, c="black", label="Observed Values")
    plt.legend(loc="upper left", prop={"size": 9.5})
    plt.tight_layout()
    plt.savefig(
        "confidence_plots_low_res/"
        + args.device
        + "_"
        + args.search_space
        + "_"
        + args.objective
        + ".pdf",
        dpi=2,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="v100")
    parser.add_argument("--objective", type=str, default="energies")
    parser.add_argument("--search_space", type=str, default="s")
    args = parser.parse_args()
    devices = (
        "a100",
        "a40",
        "h100",
        "rtx2080",
        "rtx3080",
        "a6000",
        "v100",
        "P100",
        "cpu_xeon_silver",
        "cpu_xeon_gold",
        "cpu_amd_7502",
        "cpu_amd_7513",
        "cpu_amd_7452",
    )
    objectives = ("energies", "latencies")
    search_spaces = ("s", "m", "l")
    for device in devices:
        for objective in objectives:
            for search_space in search_spaces:
                args.device = device
                args.objective = objective
                args.search_space = search_space
                make_confidence_plots(args)
                plt.clf()
