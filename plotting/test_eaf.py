from plotting.eaf import get_empirical_attainment_surface, EmpiricalAttainmentFuncPlot
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["figure.figsize"] = (6,6)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "dotted"
plt.rcParams["font.size"] = 16
plt.rcParams["figure.autolayout"] = True



def func(X: np.ndarray) -> np.ndarray:
    f1 = np.sum(X**2, axis=-1)
    f2 = np.sum((X - 2) ** 2, axis=-1)
    return np.stack([f1, f2], axis=-1)


if __name__ == "__main__":
    dim, n_samples, n_independent_runs = 2, 100, 50
    n_runs = n_independent_runs
    X = np.random.random((n_independent_runs, n_samples, dim)) * 10 - 5
    costs = func(X)

    levels = [n_independent_runs // 4, n_independent_runs // 2, 3 * n_independent_runs // 4]
    surfs = get_empirical_attainment_surface(costs=costs, levels=levels)

    _, ax = plt.subplots()
    eaf_plot = EmpiricalAttainmentFuncPlot()
    eaf_plot.plot_surface_with_band(ax, color="red", label="random", surfs=surfs)
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("test_eaf.pdf")
    plt.clf()
    _, ax = plt.subplots()
    
    rng = np.random.default_rng(12345)
    X2 = rng.random(( n_runs , n_samples , dim ))*10-5
    costs2 = func(X2)
    stacked_costs = np.stack([costs , costs2 ])
    print(stacked_costs)
    ref_point = np.array([75 , 1029]) # problem specific
    print(stacked_costs.shape)
    eaf_plot = EmpiricalAttainmentFuncPlot(ref_point = ref_point)
    colors = ["red", "blue"]
    labels = ["Random1", "Random2"]
    eaf_plot.plot_multiple_hypervolume2d_with_band(ax ,costs_array=stacked_costs ,colors=colors ,labels=labels ,normalize =False)
    #plt.legend()
    plt.tight_layout()
    plt.savefig("test_hv.pdf")