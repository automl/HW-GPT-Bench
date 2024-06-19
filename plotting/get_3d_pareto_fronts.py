import pickle
import numpy as np
from lib.utils_norm import denormalize_energy, denormalize_latency, denormalize_ppl, normalize_ppl
import os
import matplotlib
import matplotlib.pyplot as plt
increment = 2500
methods=("RS","MOREA","LS","NSGA2","LSBO","RSBO","MOASHA","EHVI")
device_energy = "P100"
device_latency = "P100"
hvs = {}
def compute_hypervolume(front, ref):
    from pygmo import hypervolume
    hv = hypervolume(front)
    return hv.compute(ref)

def plot_pareto_frontier(
        Xs,
        Ys,
        Zs,
        maxX=False,
        maxY=False):
    '''Plotting process'''

    # if scatter:
    # plt.scatter(Xs, Ys, color=color, marker=marker)
    arch_index = find_pareto_front(np.stack((Xs, Ys, Zs)).T,
                                                  return_index=True)
    return arch_index

def find_pareto_front(Y, return_index=False):
    '''
    Find pareto front (undominated part) of the input performance data.
    '''
    if len(Y) == 0:
        return np.array([])
    sorted_indices = np.argsort(Y.T[0])
    pareto_indices = []
    for idx in sorted_indices:
        # check domination relationship
        if not (
            np.logical_and(
                (Y <= Y[idx]).all(
                    axis=1), (Y < Y[idx]).any(
                axis=1))).any():
            pareto_indices.append(idx)
    return pareto_indices
scale="l"
for method in methods:
    path = "results_gpt_baselines_3d_l/"+str(method)+"/latencies_energies/"+"mogpt_"+device_latency+"_"+device_energy+"_31415927.pickle"
    with open(path,"rb") as f:
        results = pickle.load(f)
    archs = []
    ppl = []
    energy = []
    latency = []
    ppl_normalized = []
    energy_normalized = []
    latency_normalized = []
    denormalize = False
    for j,arch in enumerate(results["configs"]):
        for k in range(len(results["hw_metric_2"][j])):
            archs.append(arch)
            if denormalize:
                ppl.append(denormalize_ppl(results["perplexity"][j][k].item(),"l"))
                energy.append(denormalize_energy(results["hw_metric_2"][j][k].item(),scale="l",device=device_energy,surrogate="",data_type="",metric="energies"))
                latency.append(denormalize_latency(results["hw_metric_1"][j][k].item(),scale="l",device=device_latency,surrogate="",data_type="",metric="latencies"))
                ppl_normalized.append(results["perplexity"][j][k].item())
                energy_normalized.append(results["hw_metric_2"][j][k].item())
                latency_normalized.append(results["hw_metric_1"][j][k].item())
            else:
                ppl_c = denormalize_ppl(results["perplexity"][j][k].item(),"l")
                ppl_max_min_norm = normalize_ppl(ppl_c,method="max-min",scale="l")
                ppl.append(ppl_max_min_norm)
                energy.append(results["hw_metric_2"][j][k].item())
                latency.append(results["hw_metric_1"][j][k].item())
                ppl_normalized.append(results["perplexity"][j][k].item())
                energy_normalized.append(results["hw_metric_2"][j][k].item())
                latency_normalized.append(results["hw_metric_1"][j][k].item())

    pareto_front = plot_pareto_frontier(ppl, energy, latency)
    print("Length of pareto front: ",len(pareto_front))
    print(pareto_front)
    pareto_archs = [archs[i] for i in pareto_front]
    pareto_ppl = [ppl[i] for i in pareto_front]
    #pareto_ppl_normalized = [ppl_normalized[i] for i in pareto_front]
    pareto_energy = [energy[i] for i in pareto_front]
    #pareto_energy_normalized = [energy_normalized[i] for i in pareto_front]
    pareto_latency = [latency[i] for i in pareto_front]
    #pareto_latency_normalized = [latency_normalized[i] for i in pareto_front]
    pareto_front = {"archs":pareto_archs,"ppl":pareto_ppl,"energy":pareto_energy,"latency":pareto_latency}
    #pareto_front_normalized = {"archs":pareto_archs,"ppl":pareto_ppl_normalized,"energy":pareto_energy_normalized,"latency":pareto_latency_normalized}
    print("Length of pareto front: ",len(pareto_front["archs"]))
    with open("pareto/pareto_"+method+"_"+device_latency+"_"+device_energy+".pkl","wb") as f:
        pickle.dump(pareto_front,f)
    #with open("pareto/pareto_normalized_"+method+"_"+device_latency+"_"+device_energy+".pkl","wb") as f:
    #    pickle.dump(pareto_front_normalized,f)
    hvs[method] = compute_hypervolume(np.stack((pareto_ppl,pareto_energy,pareto_latency)).T, [1.5,1.5,1.5])
print(hvs)   