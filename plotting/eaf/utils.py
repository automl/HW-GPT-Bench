from __future__ import annotations

from fast_pareto.pareto import _change_directions

import matplotlib.pyplot as plt

import numpy as np


LOGEPS = 1e-300


def _compute_hypervolume2d(costs_array: np.ndarray, ref_point: np.ndarray) -> np.ndarray:
    # costs must be better when they are smaller
    if not np.all(costs_array[..., 0] <= ref_point[0]) or not np.all(costs_array[..., 1] <= ref_point[1]):
        raise ValueError("all values in costs must be smaller than ref_point")

    # Sort by x in asc, then by y in desc
    (n_runs, _, _) = costs_array.shape
    orders = np.lexsort((-costs_array[..., 1], costs_array[..., 0]), axis=-1)

    sorted_costs_array = np.stack([costs[order] for i, (costs, order) in enumerate(zip(costs_array, orders))])
    assert sorted_costs_array.shape == costs_array.shape

    w = np.hstack([sorted_costs_array[:, 1:, 0], np.full((n_runs, 1), ref_point[0])]) - sorted_costs_array[..., 0]
    h = ref_point[1] - np.minimum.accumulate(sorted_costs_array[..., 1], axis=-1)
    return np.sum(w * h, axis=-1)


def _change_scale(ax: plt.Axes, log_scale: list[int] | None) -> None:
    log_scale = [] if log_scale is None else log_scale
    if 0 in log_scale:
        ax.set_xscale("log")
    if 1 in log_scale:
        ax.set_yscale("log")


def _get_slighly_expanded_value_range(
    costs: np.ndarray,
    log_scale: list[int] | None = None,
) -> tuple[float, float, float, float]:
    X = costs[..., 0].flatten()
    Y = costs[..., 1].flatten()
    log_scale = log_scale if log_scale is not None else []
    x_is_log, y_is_log = 0 in log_scale, 1 in log_scale

    X = X[np.isfinite(X) & (X > LOGEPS) if x_is_log else np.isfinite(X)]
    Y = Y[np.isfinite(Y) & (Y > LOGEPS) if y_is_log else np.isfinite(Y)]
    (x_min, x_max) = (np.log(X.min()), np.log(X.max())) if x_is_log else (X.min(), X.max())
    (y_min, y_max) = (np.log(Y.min()), np.log(Y.max())) if y_is_log else (Y.min(), Y.max())

    x_min -= 0.1 * (x_max - x_min)
    x_max += 0.1 * (x_max - x_min)
    y_min -= 0.1 * (y_max - y_min)
    y_max += 0.1 * (y_max - y_min)
    (x_min, x_max) = (np.exp(x_min), np.exp(x_max)) if x_is_log else (x_min, x_max)
    (y_min, y_max) = (np.exp(y_min), np.exp(y_max)) if y_is_log else (y_min, y_max)
    return x_min, x_max, y_min, y_max


def pareto_front_to_surface(
    pareto_front: np.ndarray,
    larger_is_better_objectives: list[int] | None = None,
    log_scale: list[int] | None = None,
    x_min: float = -np.inf,
    x_max: float = np.inf,
    y_min: float = -np.inf,
    y_max: float = np.inf,
) -> np.ndarray:
    """
    Convert Pareto front set to a surface array.

    Args:
        pareto_front (np.ndarray):
            The raw pareto front solution with the shape of (n_sols, n_obj).
        larger_is_better_objectives (list[int] | None):
            The indices of the objectives that are better when the values are larger.
            If None, we consider all objectives are better when they are smaller.
        log_scale (list[int] | None):
            The indices of the log scale.
            For example, if you would like to plot the first objective in the log scale,
            you need to feed log_scale=[0].
            In principle, log_scale changes the minimum value of the axes
            from -np.inf to a small positive value.
        x_min (float):
            The lower bound of the first objective.
        x_max (float):
            The upper bound of the first objective.
        y_min (float):
            The er bound of the second objective.
        y_max (float):
            The er bound of the second objective.

    Returns:
        modified_pareto_front (np.ndarray):
            The modified pareto front solution with the shape of (n_sols + 2, n_obj).
            The head and tail of this array is now the specified bounds.
            Also this array is now sorted with respect to the first objective.
    """
    if len(pareto_front.shape) != 2:
        raise ValueError(f"The shape of pareto_front must be (n_sols, n_obj), but got {pareto_front.shape}")
    if (
        not np.all(x_min <= pareto_front[:, 0])
        or not np.all(pareto_front[:, 0] <= x_max)
        or not np.all(y_min <= pareto_front[:, 1])
        or not np.all(pareto_front[:, 1] <= y_max)
    ):
        raise ValueError(
            f"pareto_front must be in for [{x_min}, {x_max}] the first objective and "
            f"[{y_min}, {y_max}] for the second objective, but got {pareto_front}"
        )

    log_scale = [] if log_scale is None else log_scale
    larger_is_better_objectives = [] if larger_is_better_objectives is None else larger_is_better_objectives
    x_min = max(LOGEPS, x_min) if 0 in log_scale else x_min
    maximize_x = 0 in larger_is_better_objectives
    y_min = max(LOGEPS, y_min) if 1 in log_scale else y_min
    maximize_y = 1 in larger_is_better_objectives

    (n_sols, n_obj) = pareto_front.shape

    modified_pf = np.empty((n_sols + 2, n_obj))
    modified_pf[1:-1] = pareto_front
    x_border = pareto_front[:, 0].max() if maximize_x else pareto_front[:, 0].min()
    modified_pf[0] = [x_border, y_min] if maximize_y else [x_border, y_max]
    y_border = pareto_front[:, 1].max() if maximize_y else pareto_front[:, 1].min()
    modified_pf[-1] = [x_min, y_border] if maximize_x else [x_max, y_border]

    if len(larger_is_better_objectives) > 0:
        modified_pf = _change_directions(modified_pf, larger_is_better_objectives=larger_is_better_objectives)

    # Sort by the first objective, then the second objective
    order = np.lexsort((-modified_pf[:, 1], modified_pf[:, 0]))
    modified_pf = modified_pf[order]

    if len(larger_is_better_objectives) > 0:
        modified_pf = _change_directions(modified_pf, larger_is_better_objectives=larger_is_better_objectives)
    if maximize_x:
        modified_pf = np.flip(modified_pf, axis=0)

    return modified_pf


def _check_surface(surf: np.ndarray) -> np.ndarray:
    if len(surf.shape) != 2:
        raise ValueError(f"The shape of surf must be (n_points, n_obj), but got {surf.shape}")

    X = surf[:, 0]
    if np.any(np.maximum.accumulate(X) != X):
        raise ValueError("The axis [:, 0] of surf must be an increasing sequence")


def _step_direction(larger_is_better_objectives: list[int] | None) -> str:
    """
    Check here:
        https://matplotlib.org/stable/gallery/lines_bars_and_markers/step_demo.html#sphx-glr-gallery-lines-bars-and-markers-step-demo-py

    min x min (post)
        o...       R
           :
           o...
              :
              o

    max x max (pre)
        o
        :
        ...o
           :
    R      ...o

    min x max (post)
              o
              :
           o...
           :
        o...       R

    max x min (pre)
    R      ...o
           :
        ...o
        :
        o
    """
    if larger_is_better_objectives is None:
        larger_is_better_objectives = []

    large_f1_is_better = bool(0 in larger_is_better_objectives)
    return "pre" if large_f1_is_better else "post"
