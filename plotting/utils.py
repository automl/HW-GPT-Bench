from typing import Union, Tuple, List, Any, NoReturn
import pathlib

import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import pickle


def get_calibration_metrics(args):
    if (
        args.device == "v100"
        and args.search_space == "m"
        and args.objective == "energies"
    ):
        ag_path = "gpt_energies_m_v100/calibration_metrics.pkl"
        xgb_path = (
            "ensemble_models/ensemble_ensemble_xgb_m_v100_energies_callibration.pkl"
        )
        mix_path = (
            "ensemble_models/ensemble_ensemble_mix_m_v100_energies_callibration.pkl"
        )
        lightgbm_path = "ensemble_models/ensemble_ensemble_lightgbm_m_v100_energies_callibration.pkl"
    elif (
        args.device == "cpu_xeon_silver"
        and args.search_space == "m"
        and args.objective == "latencies"
    ):
        ag_path = "gpt_latencies_m_cpu_xeon_silver/calibration_metrics.pkl"
        xgb_path = "ensemble_models/ensemble_ensemble_xgb_m_cpu_xeon_silver_latencies_callibration.pkl"
        mix_path = "ensemble_models/ensemble_ensemble_mix_m_cpu_xeon_silver_latencies_callibration.pkl"
        lightgbm_path = "ensemble_models/ensemble_ensemble_lightgbm_m_cpu_xeon_silver_latencies_callibration.pkl"
    elif (
        args.device == "rtx2080"
        and args.search_space == "l"
        and args.objective == "energies"
    ):
        ag_path = "gpt_energies_l_rtx2080/calibration_metrics.pkl"
        xgb_path = (
            "ensemble_models/ensemble_ensemble_xgb_l_rtx2080_energies_callibration.pkl"
        )
        mix_path = (
            "ensemble_models/ensemble_ensemble_mix_l_rtx2080_energies_callibration.pkl"
        )
        lightgbm_path = "ensemble_models/ensemble_ensemble_lightgbm_l_rtx2080_energies_callibration.pkl"
    elif (
        args.device == "h100"
        and args.search_space == "l"
        and args.objective == "latencies"
    ):
        ag_path = "gpt_latencies_l_h100/calibration_metrics.pkl"
        xgb_path = (
            "ensemble_models/ensemble_ensemble_xgb_l_h100_latencies_callibration.pkl"
        )
        mix_path = (
            "ensemble_models/ensemble_ensemble_mix_l_h100_latencies_callibration.pkl"
        )
        lightgbm_path = "ensemble_models/ensemble_ensemble_lightgbm_l_h100_latencies_callibration.pkl"
    elif "cpu" in args.device and args.objective == "latencies":
        ag_path = (
            "gpt_latencies_"
            + args.search_space
            + "_"
            + args.device
            + "/calibration_metrics.pkl"
        )
        xgb_path = (
            "ensemble_models/ensemble_ensemble_xgb_"
            + args.search_space
            + "_"
            + args.device
            + "_latencies_callibration.pkl"
        )
        mix_path = (
            "ensemble_models/ensemble_ensemble_mix_"
            + args.search_space
            + "_"
            + args.device
            + "_latencies_callibration.pkl"
        )
        lightgbm_path = (
            "ensemble_models/ensemble_ensemble_lightgbm_"
            + args.search_space
            + "_"
            + args.device
            + "_latencies_callibration.pkl"
        )
    else:
        ag_path = (
            "gpt_"
            + args.objective
            + "_"
            + args.search_space
            + "_"
            + args.device
            + "_log/calibration_metrics.pkl"
        )
        xgb_path = (
            "ensemble_models/ensemble_ensemble_xgb_"
            + args.search_space
            + "_"
            + args.device
            + "_"
            + args.objective
            + "_callibration.pkl"
        )
        mix_path = (
            "ensemble_models/ensemble_ensemble_mix_"
            + args.search_space
            + "_"
            + args.device
            + "_"
            + args.objective
            + "_callibration.pkl"
        )
        lightgbm_path = (
            "ensemble_models/ensemble_ensemble_lightgbm_"
            + args.search_space
            + "_"
            + args.device
            + "_"
            + args.objective
            + "_callibration.pkl"
        )
    with open(ag_path, "rb") as f:
        a_autogluon = pickle.load(f)
    with open(xgb_path, "rb") as f:
        a_xgb = pickle.load(f)
    with open(mix_path, "rb") as f:
        a_mix = pickle.load(f)
    with open(lightgbm_path, "rb") as f:
        a_lightgbm = pickle.load(f)
    return a_autogluon, a_xgb, a_mix, a_lightgbm


def filter_subset(input_list: List[List[Any]], n_subset: int) -> List[List[Any]]:
    """Keep only n_subset random indices from all lists given in input_list.

    Args:
        input_list: list of lists.
        n_subset: Number of points to plot after filtering.

    Returns:
        List of all input lists with sizes reduced to n_subset.
    """
    assert type(n_subset) is int
    n_total = len(input_list[0])
    print(n_total)
    print(n_subset)
    idx = np.random.choice(range(n_total), n_subset, replace=False)
    idx = np.sort(idx)
    output_list = []
    for inp in input_list:
        outp = inp[idx]
        output_list.append(outp)
    return output_list


from uncertainty_toolbox.metrics_calibration import (
    get_proportion_lists,
    get_proportion_lists_vectorized,
    adversarial_group_calibration,
    miscalibration_area,
    miscalibration_area_from_proportions,
)


def plot_calibration(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    n_subset: Union[int, None] = None,
    curve_label: Union[str, None] = None,
    vectorized: bool = True,
    exp_props: Union[np.ndarray, None] = None,
    obs_props: Union[np.ndarray, None] = None,
    ax: Union[matplotlib.axes.Axes, None] = None,
    prop_type: str = "interval",
    color: str = "blue",
    ideal_flag: bool = True,
    device: str = "RTX2080",
) -> matplotlib.axes.Axes:
    """Plot the observed proportion vs prediction proportion of outputs falling into a
    range of intervals, and display miscalibration area.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        n_subset: Number of points to plot after filtering.
        curve_label: legend label str for calibration curve.
        vectorized: plot using get_proportion_lists_vectorized.
        exp_props: plot using the given expected proportions.
        obs_props: plot using the given observed proportions.
        ax: matplotlib.axes.Axes object.
        prop_type: "interval" to measure observed proportions for centered prediction intervals,
                   and "quantile" for observed proportions below a predicted quantile.
                   Ignored if exp_props and obs_props are provided as inputs.

    Returns:
        matplotlib.axes.Axes object with plot added.
    """
    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Optionally select a subset
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)

    if (exp_props is None) or (obs_props is None):
        # Compute exp_proportions and obs_proportions
        if vectorized:
            (
                exp_proportions,
                obs_proportions,
            ) = get_proportion_lists_vectorized(
                y_pred, y_std, y_true, prop_type=prop_type
            )
        else:
            (exp_proportions, obs_proportions) = get_proportion_lists(
                y_pred, y_std, y_true, prop_type=prop_type
            )
    else:
        # If expected and observed proportions are given
        exp_proportions = np.array(exp_props).flatten()
        obs_proportions = np.array(obs_props).flatten()
        if exp_proportions.shape != obs_proportions.shape:
            raise RuntimeError("exp_props and obs_props shape mismatch")

    # Set label
    if curve_label is None:
        curve_label = "Predictor"

    # Plot
    if ideal_flag:
        ax.plot([0, 1], [0, 1], "--", label="Ideal", c="black")
    ax.plot(exp_proportions, obs_proportions, label=curve_label, c=color)
    ax.fill_between(exp_proportions, exp_proportions, obs_proportions, alpha=0.2)

    # Format plot
    ax.set_xlabel("Predicted Proportion in Interval")
    ax.set_ylabel("Observed Proportion in Interval")
    ax.axis("square")

    buff = 0.01
    ax.set_xlim([0 - buff, 1 + buff])
    ax.set_ylim([0 - buff, 1 + buff])

    ax.set_title("Average Calibration on " + str(device))

    # Compute miscalibration area
    miscalibration_area = miscalibration_area_from_proportions(
        exp_proportions=exp_proportions, obs_proportions=obs_proportions
    )

    # Annotate plot with the miscalibration area
    """ax.text(
        x=0.95,
        y=0.05,
        s="Miscalibration area = %.2f" % miscalibration_area,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize="small",
    )"""

    return ax


def plot_intervals_ordered(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    n_subset: Union[int, None] = None,
    ylims: Union[Tuple[float, float], None] = None,
    num_stds_confidence_bound: int = 2,
    ax: Union[matplotlib.axes.Axes, None] = None,
    color: str = "blue",
    device: str = "Xeon Silver",
    label: str = "",
    true_flag: bool = False,
    metric: str = "Latency",
) -> matplotlib.axes.Axes:
    """Plot predictions and predictive intervals versus true values, with points ordered
    by true value along x-axis.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        n_subset: Number of points to plot after filtering.
        ylims: a tuple of y axis plotting bounds, given as (lower, upper).
        num_stds_confidence_bound: width of intervals, in terms of number of standard
            deviations.
        ax: matplotlib.axes.Axes object.

    Returns:
        matplotlib.axes.Axes object with plot added.
    """
    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Optionally select a subset
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)

    order = np.argsort(y_true.flatten())
    y_pred, y_std, y_true = y_pred[order], y_std[order], y_true[order]
    xs = np.arange(len(order))
    intervals = num_stds_confidence_bound * y_std
    ylims = [min(y_true) - 0.0001, max(y_true) + 0.0001]
    # Plot
    ax.errorbar(
        xs,
        y_pred,
        intervals,
        fmt="o",
        ls="none",
        linewidth=1.5,
        c=color,
        alpha=0.5,
        markersize=4,
    )
    h1 = plt.plot(xs, y_pred, "o", c=color, alpha=1, label=label, markersize=4)
    if true_flag:
        h2 = plt.plot(
            xs, y_true, "--", linewidth=3.0, c="black", label="Observed " + str(metric)
        )

    # Legend
    # if true_flag:
    # plt.legend([h1[0], h2[0]], ["Predicted Values "+label, "Observed Values "+device], loc=4)
    # else:
    # plt.legend([h1[0]], ["Predicted Values "+label], loc=4)

    # Determine lims
    if ylims is None:
        intervals_lower_upper = [y_pred - intervals, y_pred + intervals]
        lims_ext = [
            int(np.floor(np.min(intervals_lower_upper[0]))),
            int(np.ceil(np.max(intervals_lower_upper[1]))),
        ]
    else:
        lims_ext = ylims

    # Format plot
    ax.set_ylim(lims_ext)
    ax.set_xlabel("Index (Ordered by Observed Value)")
    ax.set_ylabel("Predicted " + str(metric) + " and Intervals")
    ax.set_title("Ordered Prediction Intervals " + device)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")
    return ax


from typing import Union, Tuple, List, Any, NoReturn
import pathlib

import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt


def filter_subset(input_list: List[List[Any]], n_subset: int) -> List[List[Any]]:
    """Keep only n_subset random indices from all lists given in input_list.

    Args:
        input_list: list of lists.
        n_subset: Number of points to plot after filtering.

    Returns:
        List of all input lists with sizes reduced to n_subset.
    """
    assert type(n_subset) is int
    n_total = len(input_list[0])
    print(n_total)
    print(n_subset)
    idx = np.random.choice(range(n_total), n_subset, replace=False)
    idx = np.sort(idx)
    output_list = []
    for inp in input_list:
        outp = inp[idx]
        output_list.append(outp)
    return output_list


from uncertainty_toolbox.metrics_calibration import (
    get_proportion_lists,
    get_proportion_lists_vectorized,
    adversarial_group_calibration,
    miscalibration_area,
    miscalibration_area_from_proportions,
)


def plot_xy(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    x: np.ndarray,
    n_subset: Union[int, None] = None,
    ylims: Union[Tuple[float, float], None] = None,
    xlims: Union[Tuple[float, float], None] = None,
    num_stds_confidence_bound: int = 2,
    leg_loc: Union[int, str] = 3,
    ax: Union[matplotlib.axes.Axes, None] = None,
    color: str = "blue",
    plot_true: bool = True,
    method: str = "AutoGluon",
    device="Xeon Silver (Lat)",
    metric="Latency (ms)",
) -> matplotlib.axes.Axes:
    """Plot one-dimensional inputs with associated predicted values, predictive
    uncertainties, and true values.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        x: 1D array of input values for the held out dataset.
        n_subset: Number of points to plot after filtering.
        ylims: a tuple of y axis plotting bounds, given as (lower, upper).
        xlims: a tuple of x axis plotting bounds, given as (lower, upper).
        num_stds_confidence_bound: width of confidence band, in terms of number of
            standard deviations.
        leg_loc: location of legend as a str or legend code int.
        ax: matplotlib.axes.Axes object.

    Returns:
        matplotlib.axes.Axes object with plot added.
    """
    # Create ax if it doesn't exist

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Order points in order of increasing x
    order = np.argsort(x)
    y_pred, y_std, y_true, x = (
        y_pred[order],
        y_std[order],
        y_true[order],
        x[order],
    )

    # Optionally select a subset
    if n_subset is not None:
        [y_pred, y_std, y_true, x] = filter_subset([y_pred, y_std, y_true, x], n_subset)
    ylims = [min(y_true) - 0.0001, max(y_true) + 0.0001]
    intervals = num_stds_confidence_bound * y_std
    if plot_true:
        h1 = ax.plot(x, y_true, ".", mec="black", mfc="None", label="Observations")
    h2 = ax.plot(
        x, y_pred, "-", c=color, linewidth=1, label="Predictions " + str(method)
    )
    h3 = ax.fill_between(
        x,
        y_pred - intervals,
        y_pred + intervals,
        color=color,
        alpha=0.2,
        label="$95\%$ Interval " + str(method),
    )

    # Format plot
    if ylims is not None:
        ax.set_ylim(ylims)

    if xlims is not None:
        ax.set_xlim(xlims)

    ax.set_xlabel("Arch (1D representation)")
    ax.set_ylabel(metric)
    ax.set_title("Confidence Band " + device)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")

    return ax
