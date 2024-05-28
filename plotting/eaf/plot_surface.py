from __future__ import annotations

from copy import deepcopy
from typing import Any

from eaf.utils import (
    LOGEPS,
    _change_scale,
    _check_surface,
    _compute_hypervolume2d,
    _get_slighly_expanded_value_range,
    _step_direction,
    pareto_front_to_surface,
)

from fast_pareto.pareto import _change_directions

import matplotlib
import matplotlib.pyplot as plt

import numpy as np


def _extract_marker_kwargs(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    marker_kwargs: dict[str, Any] = dict()
    new_kwargs: dict[str, Any] = dict()
    for k, v in kwargs.items():
        if k.startswith("mark"):
            marker_kwargs[k] = v
        else:
            new_kwargs[k] = v

    return marker_kwargs, new_kwargs


class EmpiricalAttainmentFuncPlot:
    """
    The class to plot empirical attainment function.

    Args:
        true_pareto_sols (np.ndarray | None):
            The true Pareto solutions if available.
            It is used to compute the best values of each objective.
        larger_is_better_objectives (list[int] | None):
            The indices of the objectives that are better when the values are larger.
            If None, we consider all objectives are better when they are smaller.
        log_scale (list[int] | None):
            The indices of the log scale.
            For example, if you would like to plot the first objective in the log scale,
            you need to feed log_scale=[0].
            In principle, log_scale changes the minimum value of the axes
            from -np.inf to a small positive value.
        x_min, x_max, y_min, y_max (float):
            The lower/upper bounds for each objective if available.
            It is used to fix the value ranges of each objective in plots.
        ref_point (np.ndarray | None):
            The reference point to be used for the visualization of hypervolume plots.
    """

    def __init__(
        self,
        true_pareto_sols: np.ndarray | None = None,
        larger_is_better_objectives: list[int] | None = None,
        log_scale: list[int] | None = None,
        x_min: float = np.inf,
        x_max: float = -np.inf,
        y_min: float = np.inf,
        y_max: float = -np.inf,
        ref_point: np.ndarray | None = None,
    ):
        self.step_dir = _step_direction(larger_is_better_objectives)
        self.larger_is_better_objectives = (
            larger_is_better_objectives if larger_is_better_objectives is not None else []
        )
        self._ref_point = ref_point.copy().astype(np.float64) if ref_point is not None else None
        self.log_scale = log_scale if log_scale is not None else []
        self.x_is_log, self.y_is_log = 0 in self.log_scale, 1 in self.log_scale
        self._plot_kwargs = dict(
            larger_is_better_objectives=larger_is_better_objectives,
            log_scale=log_scale,
        )

        if true_pareto_sols is not None:
            self.x_min, self.x_max, self.y_min, self.y_max = _get_slighly_expanded_value_range(
                true_pareto_sols, self.log_scale
            )
            self._true_pareto_sols = true_pareto_sols.copy()
        else:
            # We cannot plot until we call _transform_surface_list
            self._true_pareto_sols = None
            self.x_min = max(LOGEPS, x_min) if 0 in self.log_scale else x_min
            self.x_max = x_max
            self.y_min = max(LOGEPS, y_min) if 1 in self.log_scale else y_min
            self.y_max = y_max

    def _transform_surface_list(self, surfs_list: list[np.ndarray]) -> list[np.ndarray]:
        for surf in surfs_list:
            x_min, x_max, y_min, y_max = _get_slighly_expanded_value_range(surf, self.log_scale)
            self.x_min, self.x_max = min(self.x_min, x_min), max(self.x_max, x_max)
            self.y_min, self.y_max = min(self.y_min, y_min), max(self.y_max, y_max)

        for surf in surfs_list:
            lb = LOGEPS if self.x_is_log else -np.inf
            surf[..., 0][surf[..., 0] == lb] = self.x_min
            surf[..., 0][surf[..., 0] == np.inf] = self.x_max

            lb = LOGEPS if self.y_is_log else -np.inf
            surf[..., 1][surf[..., 1] == lb] = self.y_min
            surf[..., 1][surf[..., 1] == np.inf] = self.y_max

        return surfs_list

    def plot_surface(
        self,
        ax: plt.Axes,
        surf: np.ndarray,
        color: str,
        label: str,
        linestyle: str | None = None,
        marker: str | None = None,
        transform: bool = True,
        **kwargs: Any,
    ) -> matplotlib.lines.Line2D:
        """
        Plot multiple surfaces.

        Args:
            ax (plt.Axes):
                The subplots axes.
            surf (np.ndarray):
                The vertices of the empirical attainment surface.
                The shape must be (X.size, 2).
            color (str):
                The color of the plot.
            label (str):
                The label of the plot.
            linestyle (str | None):
                The linestyle of the plot.
            marker (str | None):
                The marker of the plot.
            transform (bool):
                Whether automatically transforming based on (x_min, x_max) and (y_min, y_max).
            kwargs:
                The kwargs for ax.plot.

        Returns:
            line (matplotlib.lines.Line2D):
                The plotted line object.
        """
        if len(surf.shape) != 2 or surf.shape[1] != 2:
            raise ValueError(f"The shape of surf must be (n_points, 2), but got {surf.shape}")

        _surf = surf.copy()
        if transform:
            _surf = self._transform_surface_list(surfs_list=[_surf])[0]
            ax.set_xlim(self.x_min, self.x_max)
            ax.set_ylim(self.y_min, self.y_max)

        kwargs.update(drawstyle=f"steps-{self.step_dir}")
        _check_surface(_surf)
        X, Y = _surf[:, 0], _surf[:, 1]
        line = ax.plot(X, Y, color=color, label=label, linestyle=linestyle, marker=marker, **kwargs)[0]
        _change_scale(ax, self.log_scale)
        return line

    def plot_true_pareto_surface(
        self,
        ax: plt.Axes,
        color: str,
        label: str,
        linestyle: str | None = None,
        marker: str | None = None,
        **kwargs: Any,
    ) -> matplotlib.lines.Line2D:
        """
        Plot the true pareto front surface.
        The true pareto front must be provided when instantiating this class.

        Args:
            ax (plt.Axes):
                The subplots axes.
            color (str):
                The color of the plot.
            label (str):
                The label of the plot.
            linestyle (str | None):
                The linestyle of the plot.
            marker (str | None):
                The marker of the plot.
            kwargs:
                The kwargs for ax.plot.

        Returns:
            line (matplotlib.lines.Line2D):
                The plotted line object.
        """
        if self._true_pareto_sols is None:
            raise AttributeError("true_pareto_sols is not provided at the instantiation")

        true_pareto_surf = pareto_front_to_surface(
            self._true_pareto_sols.copy(),
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            **self._plot_kwargs,
        )
        return self.plot_surface(
            ax,
            surf=true_pareto_surf,
            color=color,
            label=label,
            linestyle=linestyle,
            marker=marker,
            transform=False,
            **kwargs,
        )

    def plot_multiple_surface(
        self,
        ax: plt.Axes,
        surfs: np.ndarray | list[np.ndarray],
        colors: list[str],
        labels: list[str],
        linestyles: list[str | None] | None = None,
        markers: list[str | None] | None = None,
        **kwargs: Any,
    ) -> list[matplotlib.lines.Line2D]:
        """
        Plot multiple surfaces.

        Args:
            ax (plt.Axes):
                The subplots axes.
            surfs (np.ndarray | list[np.ndarray]):
                The vertices of the empirical attainment surfaces for each plot.
                Each element should have the shape of (X.size, 2).
                If this is an array, then the shape must be (n_surf, X.size, 2).
            colors (list[str]):
                The colors of each plot
            labels (list[str]):
                The labels of each plot.
            linestyles (list[str | None] | None):
                The linestyles of each plot.
            markers (list[str | None] | None):
                The markers of each plot.
            kwargs:
                The kwargs for ax.plot.

        Returns:
            lines (list[matplotlib.lines.Line2D]):
                A list of the plotted line objects.
        """
        lines: list[matplotlib.lines.Line2D] = []
        _surfs = deepcopy(surfs)
        _surfs = self._transform_surface_list(_surfs)

        n_surfs = len(_surfs)
        linestyles = linestyles if linestyles is not None else [None] * n_surfs
        markers = markers if markers is not None else [None] * n_surfs
        for surf, color, label, linestyle, marker in zip(_surfs, colors, labels, linestyles, markers):
            kwargs.update(color=color, label=label, linestyle=linestyle, marker=marker)
            line = self.plot_surface(ax, surf, transform=False, **kwargs)
            lines.append(line)

        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        return lines

    def plot_surface_with_band(
        self,
        ax: plt.Axes,
        surfs: np.ndarray,
        color: str,
        label: str,
        linestyle: str | None = None,
        marker: str | None = None,
        transform: bool = True,
        **kwargs: Any,
    ) -> matplotlib.lines.Line2D:
        """
        Plot the surface with a band.
        Typically, we would like to plot median with the band between
        25% -- 75% percentile attainment surfaces.

        Args:
            ax (plt.Axes):
                The subplots axes.
            surfs (np.ndarray):
                The vertices of the empirical attainment surfaces for each level.
                If surf[i, j, 1] takes np.inf, this is not actually on the surface.
                The shape is (3, X.size, 2).
            color (str):
                The color of the plot
            label (str):
                The label of the plot.
            linestyle (str | None):
                The linestyle of the plot.
            marker (str | None):
                The marker of the plot.
            transform (bool):
                Whether automatically transforming based on (x_min, x_max) and (y_min, y_max).
            kwargs:
                The kwargs for ax.plot.

        Returns:
            line (matplotlib.lines.Line2D):
                The plotted line object.
        """
        if surfs.shape[0] != 3:
            raise ValueError(f"plot_surface_with_band requires three levels, but got only {surfs.shape[0]} levels")

        _surfs = deepcopy(surfs)
        if transform:
            _surfs = self._transform_surface_list(surfs_list=_surfs)
            ax.set_xlim(self.x_min, self.x_max)
            ax.set_ylim(self.y_min, self.y_max)

        for surf in _surfs:
            _check_surface(surf)

        X = _surfs[0, :, 0]

        marker_kwargs, kwargs = _extract_marker_kwargs(**kwargs)
        kwargs.update(color=color)
        alpha = kwargs.pop("alpha", None)
        ax.fill_between(X, _surfs[0, :, 1], _surfs[2, :, 1], alpha=0.2, step=self.step_dir, **kwargs)
        kwargs["alpha"] = alpha

        # marker and linestyle are only for plot
        kwargs.update(label=label, linestyle=linestyle, marker=marker, **marker_kwargs)
        line = ax.plot(X, _surfs[1, :, 1], drawstyle=f"steps-{self.step_dir}", **kwargs)[0]
        _change_scale(ax, self.log_scale)
        return line

    def plot_multiple_surface_with_band(
        self,
        ax: plt.Axes,
        surfs_list: np.ndarray | list[np.ndarray],
        colors: list[str],
        labels: list[str],
        linestyles: list[str | None] | None = None,
        markers: list[str | None] | None = None,
        **kwargs: Any,
    ) -> list[matplotlib.lines.Line2D]:
        """
        Plot multiple surfaces with a band.

        Args:
            ax (plt.Axes):
                The subplots axes.
            surfs_list (np.ndarray | list[np.ndarray]):
                The vertices of the empirical attainment surfaces for each plot.
                Each element should have the shape of (3, X.size, 2).
                If this is an array, then the shape must be (n_surf, 3, X.size, 2).
            colors (list[str]):
                The colors of each plot.
            labels (list[str]):
                The labels of each plot.
            linestyles (list[str | None] | None):
                The linestyles of each plot.
            markers (list[str | None] | None):
                The markers of each plot.
            kwargs:
                The kwargs for ax.plot.

        Returns:
            lines (list[matplotlib.lines.Line2D]):
                A list of the plotted line objects.
        """
        lines: list[matplotlib.lines.Line2D] = []
        _surfs_list = deepcopy(surfs_list)
        _surfs_list = self._transform_surface_list(_surfs_list)

        n_surfs = len(_surfs_list)
        linestyles = linestyles if linestyles is not None else [None] * n_surfs
        markers = markers if markers is not None else [None] * n_surfs
        for surf, color, label, linestyle, marker in zip(_surfs_list, colors, labels, linestyles, markers):
            kwargs.update(color=color, label=label, linestyle=linestyle, marker=marker)
            line = self.plot_surface_with_band(ax, surf, transform=False, **kwargs)
            lines.append(line)

        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        return lines

    def _transform_ref_point_and_costs_array(self, costs_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ref_point = self._ref_point.copy()
        _costs_array = costs_array.copy()
        if self.x_is_log:
            _costs_array[..., 0] = np.log(_costs_array[..., 0])
            ref_point[0] = np.log(ref_point[0])
        if self.y_is_log:
            _costs_array[..., 1] = np.log(_costs_array[..., 1])
            ref_point[1] = np.log(ref_point[1])

        if len(self.larger_is_better_objectives) > 0:
            _costs_array = _change_directions(_costs_array, self.larger_is_better_objectives)
            ref_point = _change_directions(ref_point[np.newaxis], self.larger_is_better_objectives)[0]

        return ref_point, _costs_array

    def plot_hypervolume2d_with_band(
        self,
        ax: plt.Axes,
        costs_array: np.ndarray,
        color: str,
        label: str,
        linestyle: str | None = None,
        marker: str | None = None,
        log: bool = False,
        axis_label: bool = True,
        normalize: bool = True,
        **kwargs: Any,
    ) -> matplotlib.lines.Line2D:
        """
        Plot the hypervolume with a standard error band.

        Args:
            ax (plt.Axes):
                The subplots axes.
            costs_array (np.ndarray):
                The costs obtained in the observations.
                The shape must be (n_independent_runs, n_samples, n_obj).
                For now, we only support n_obj == 2.
            color (str):
                The color of the plot.
            label (str):
                The label of the plot.
            linestyle (str | None):
                The linestyle of the plot.
            marker (str | None):
                The marker of the plot.
            log (bool):
                Whether to use the log scale in the y-axis of the plot.
            axis_label (bool):
                Whether to add the default axis labels or not.
            normalize (bool):
                Whether to normalize the hypervolume or not.
                If True, the y-axis is scaled into [0, 1].
            kwargs:
                The kwargs for ax.plot.

        Returns:
            line (matplotlib.lines.Line2D):
                The plotted line object.
        """
        if len(costs_array.shape) != 3 or costs_array.shape[-1] != 2:
            raise ValueError(
                f"The shape of costs_array must be (n_independent_runs, n_points, 2), but got {costs_array.shape}"
            )
        if self._ref_point is None:
            raise AttributeError("ref_point must be provided for plot_hypervolume2d_with_band")

        ref_point, _costs_array = self._transform_ref_point_and_costs_array(costs_array)
        (n_runs, n_observations, _) = _costs_array.shape
        hvs = np.zeros((n_runs, n_observations))
        for i in range(n_observations):
            hvs[:, i] = _compute_hypervolume2d(costs_array=_costs_array[:, : i + 1], ref_point=ref_point)
        #print(hvs)
        if normalize:
            if self._true_pareto_sols is None:
                raise AttributeError(
                    "true_pareto_sols is not provided at the instantiation, but it is required to "
                    "call `plot_hypervolume2d_with_band` with normalize=True. Consider using normalize=False"
                )

            max_hv = self._compute_true_pareto_surface_hypervolume2d()
            hvs /= max_hv

        T = np.arange(n_observations) + 1
        m, s = np.mean(hvs, axis=0), np.std(hvs, axis=0) / np.sqrt(n_observations)

        marker_kwargs, kwargs = _extract_marker_kwargs(**kwargs)
        kwargs.update(color=color)
        alpha = kwargs.pop("alpha", None)
        ax.fill_between(T, m - s, m + s, alpha=0.2, **kwargs)
        kwargs["alpha"] = alpha

        # marker and linestyle are only for plot
        kwargs.update(label=label, linestyle=linestyle, marker=marker, **marker_kwargs)
        line = ax.plot(T, m, **kwargs)[0]

        if log:
            ax.set_yscale("log")
        if axis_label:
            ax.set_xlabel("Number of config evaluations")
            ax.set_ylabel(("Normalized " if normalize else "") + "Hypervolume")

        ax.set_xlim((1, n_observations))
        return line

    def plot_multiple_hypervolume2d_with_band(
        self,
        ax: plt.Axes,
        costs_array: np.ndarray,
        colors: list[str],
        labels: list[str],
        linestyles: list[str | None] | None = None,
        markers: list[str | None] | None = None,
        log: bool = False,
        axis_label: bool = True,
        normalize: bool = True,
        **kwargs: Any,
    ) -> list[matplotlib.lines.Line2D]:
        """
        Plot multiple hypervolume curves.

        Args:
            ax (plt.Axes):
                The subplots axes.
            costs_array (np.ndarray):
                The costs obtained in the observations.
                The shape must be (n_curves, n_independent_runs, n_samples, n_obj).
                For now, we only support n_obj == 2.
            colors (list[str]):
                The colors of each plot.
            labels (list[str]):
                The labels of each plot.
            linestyles (list[str | None] | None):
                The linestyles of each plot.
            markers (list[str | None] | None):
                The markers of each plot.
            log (bool):
                Whether to use the log scale in the y-axis of the plot.
            axis_label (bool):
                Whether to add the default axis labels or not.
            normalize (bool):
                Whether to normalize the hypervolume or not.
                If True, the y-axis is scaled into [0, 1].
            kwargs:
                The kwargs for ax.plot.

        Returns:
            lines (list[matplotlib.lines.Line2D]):
                A list of the plotted line objects.
        """
        if len(costs_array.shape) != 4 or costs_array.shape[-1] != 2:
            raise ValueError(
                "The shape of costs_array must be (n_curves, n_independent_runs, n_points, 2),"
                f" but got {costs_array.shape}"
            )

        lines: list[matplotlib.lines.Line2D] = []
        n_observations = costs_array.shape[-2]
        n_lines = len(costs_array)
        linestyles = linestyles if linestyles is not None else [None] * n_lines
        markers = markers if markers is not None else [None] * n_lines
        for _costs_array, color, label, linestyle, marker in zip(costs_array, colors, labels, linestyles, markers):
            kwargs.update(color=color, label=label, linestyle=linestyle, marker=marker, normalize=normalize)
            line = self.plot_hypervolume2d_with_band(ax, _costs_array, log=False, axis_label=False, **kwargs)
            lines.append(line)

        if log:
            ax.set_yscale("log")
        if axis_label:
            ax.set_xlabel("Number of config evaluations")
            ax.set_ylabel(("Normalized " if normalize else "") + "Hypervolume")

        ax.set_xlim((1, n_observations))
        return lines

    def _compute_true_pareto_surface_hypervolume2d(self) -> float:
        if self._true_pareto_sols is None:
            raise AttributeError("true_pareto_sols is not provided at the instantiation")

        if self._ref_point is None:
            raise AttributeError("ref_point must be provided for plot_hypervolume2d_with_band")

        ref_point, true_pf = self._transform_ref_point_and_costs_array(self._true_pareto_sols)
        hv = _compute_hypervolume2d(true_pf[np.newaxis], ref_point)[0]
        return hv

    def plot_true_pareto_surface_hypervolume2d(
        self,
        ax: plt.Axes,
        n_observations: int,
        color: str,
        label: str,
        linestyle: str | None = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> matplotlib.lines.Line2D:
        """
        Plot multiple surfaces.

        Args:
            ax (plt.Axes):
                The subplots axes.
            n_observations (int):
                How many samples happened in each experiment.
                We need this value to set xlim.
            color (str):
                The color of the plot.
            label (str):
                The label of the plot.
            linestyle (str | None):
                The linestyle of the plot.
            kwargs:
                The kwargs for ax.plot.

        Returns:
            line (matplotlib.lines.Line2D):
                The plotted line object.
        """

        if normalize:
            hv = 1.0
        else:
            hv = self._compute_true_pareto_surface_hypervolume2d()

        kwargs.update(colors=color, label=label, linestyle=linestyle)
        line = ax.hlines(y=hv, xmin=1, xmax=n_observations, **kwargs)
        ax.set_xlim((1, n_observations))
        return line
