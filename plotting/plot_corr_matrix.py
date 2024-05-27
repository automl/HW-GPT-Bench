import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats_scipy
import os

devices = [
    "P100",
    "a6000",
    "rtx2080",
    "rtx3080",
    "v100",
    "a100",
    "h100",
]  # "cpu_llgpu", "cpu_alldlc", "cpu_p100", "cpu_p100", "cpu_a6000", "cpu_leta"]
devices_energy_dict = {}
increment = 2500
for device in devices:
    cat_list = []
    for i in range(0, 10000, increment):
        path = (
            "/work/dlclarge2/sukthank-hw-llm-bench/HW-Aware-LLM-Bench/latency_"
            + device
            + "/efficiency_energy_observations_tracker_"
            + str(i)
            + "_"
            + str(i + increment)
            + ".pkl"
        )
        if not os.path.exists(path):
            print(f"Path {path} does not exist")
            break
        with open(path, "rb") as f:
            a = pickle.load(f)
            cat_list.extend(a)
    if len(cat_list) == 10000:
        energies = []
        for i in cat_list:
            energies.append(np.median(i["energy_gpu"]))
        devices_energy_dict[device + "_energy"] = energies


# read latencies
def process(data, device_type="cpu"):
    new_stats = {}
    new_stats["archs"] = []
    new_stats["flops"] = []
    new_stats["macs"] = []
    new_stats["latency"] = []
    new_stats["energy"] = []
    new_stats["params"] = []
    for arch in data:
        new_stats["archs"].append(arch["arch"])
        new_stats["flops"].append(arch["flops"])
        if device_type == "cpu":
            latencies = arch["times_profiler_cpu"]
            units = arch["unit_cpu"]
            energy = arch["mean_energy_cpu"]
        else:
            latencies = arch["times_profiler_gpu"]
            units = arch["unit_gpu"]
            energy = arch["mean_energy_gpu"]
        latencies_scaled = []
        for i in range(len(latencies)):
            if units[i] == "ms":
                latencies_scaled.append(latencies[i])
            elif units[i] == "s":
                latencies_scaled.append(latencies[i] * 1000)
        new_stats["latency"].append(np.mean(latencies_scaled))
        new_stats["energy"].append(energy)
        new_stats["params"].append(arch["params"])
        new_stats["macs"].append(arch["macs"])
    return new_stats


devices = [
    "P100",
    "a6000",
    "rtx2080",
    "rtx3080",
    "v100",
    "a100",
    "h100",
    "cpu_mlgpu",
    "cpu_alldlc",
    "cpu_p100",
    "cpu_a6000",
    "cpu_meta",
]
devices_latency_dict = {}
devices_mac_dict = {}
devices_params_dict = {}
devices_flops_dict = {}
increment = 2500
for device in devices:
    cat_list = []
    for i in range(0, 10000, increment):
        path = (
            "/work/dlclarge2/sukthank-hw-llm-bench/HW-Aware-LLM-Bench/latency_"
            + device
            + "/efficiency_observations_"
            + str(i)
            + "_"
            + str(i + increment)
            + ".pkl"
        )
        with open(path, "rb") as f:
            a = pickle.load(f)
            cat_list.extend(a)
    if len(cat_list) == 10000:
        if "cpu" in device:
            stats = process(cat_list, device_type="cpu")
            devices_energy_dict[device + "_energy"] = stats["energy"]
        else:
            stats = process(cat_list, device_type="gpu")
        devices_latency_dict[device + "_latency"] = stats["latency"]
        if "flops" not in devices_flops_dict and device == "rtx2080":
            devices_mac_dict["macs"] = stats["macs"]
            devices_params_dict["params"] = stats["params"]
            devices_flops_dict["flops"] = stats["flops"]

# plot correlation matrix
# merge all dictionaries
all_dict = {}
all_dict.update(devices_latency_dict)
all_dict.update(devices_energy_dict)
all_dict.update(devices_flops_dict)
all_dict.update(devices_params_dict)
all_dict.update(devices_mac_dict)
# remove "cpu_p100_energy"
del all_dict["cpu_p100_energy"]
all_dict.update(devices_flops_dict)
# check if len of all items in dict is 10000
# for key in all_dict.keys():
#    print(len(all_dict[key]))
#    print(key)
#    assert len(all_dict[key]) == 10000
corr_mat = np.zeros((len(all_dict), len(all_dict)))
keys = list(all_dict.keys())
for i in range(len(keys)):
    for j in range(len(keys)):
        print(keys[i], keys[j])
        print(len(all_dict[keys[i]]), len(all_dict[keys[j]]))
        # print shapes
        print(all_dict[keys[j]][0:5])
        corr, _ = stats_scipy.kendalltau(all_dict[keys[i]], all_dict[keys[j]])
        # round to 2 places
        corr_mat[i, j] = round(corr, 2)
plt.figure(figsize=(20, 20))
sns.heatmap(corr_mat, annot=True, xticklabels=keys, yticklabels=keys)
# save
plt.savefig("corr_matrix.pdf")

# only cpus corr
# merge all dictionaries
all_dict = {}
all_dict.update(devices_latency_dict)
all_dict.update(devices_energy_dict)
all_dict.update(devices_flops_dict)
all_dict.update(devices_params_dict)
all_dict.update(devices_mac_dict)
# remove "cpu_p100_energy"
del all_dict["cpu_p100_energy"]
for key in list(all_dict.keys()):
    if "cpu" not in key and (key != "macs" and key != "params" and key != "flops"):
        del all_dict[key]

corr_mat = np.zeros((len(all_dict), len(all_dict)))
keys = list(all_dict.keys())
for i in range(len(keys)):
    for j in range(len(keys)):
        print(keys[i], keys[j])
        print(len(all_dict[keys[i]]), len(all_dict[keys[j]]))
        # print shapes
        print(all_dict[keys[j]][0:5])
        corr, _ = stats_scipy.kendalltau(all_dict[keys[i]], all_dict[keys[j]])
        # round to 2 places
        corr_mat[i, j] = round(corr, 2)
plt.figure(figsize=(20, 20))
sns.heatmap(corr_mat, annot=True, xticklabels=keys, yticklabels=keys)
# save
plt.savefig("corr_matrix_cpu.pdf")

# only gpus corr
# merge all dictionaries
all_dict = {}
all_dict.update(devices_latency_dict)
all_dict.update(devices_energy_dict)
all_dict.update(devices_flops_dict)
all_dict.update(devices_params_dict)
all_dict.update(devices_mac_dict)
# remove "cpu_p100_energy"
del all_dict["cpu_p100_energy"]
for key in list(all_dict.keys()):
    if "cpu" in key:
        del all_dict[key]

corr_mat = np.zeros((len(all_dict), len(all_dict)))
keys = list(all_dict.keys())
for i in range(len(keys)):
    for j in range(len(keys)):
        print(keys[i], keys[j])
        print(len(all_dict[keys[i]]), len(all_dict[keys[j]]))
        # print shapes
        print(all_dict[keys[j]][0:5])
        corr, _ = stats_scipy.kendalltau(all_dict[keys[i]], all_dict[keys[j]])
        # round to 2 places
        corr_mat[i, j] = round(corr, 2)
plt.figure(figsize=(20, 20))
sns.heatmap(corr_mat, annot=True, xticklabels=keys, yticklabels=keys)
# save
plt.savefig("corr_matrix_gpu.pdf")
