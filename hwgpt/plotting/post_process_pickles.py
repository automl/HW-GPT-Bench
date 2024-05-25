import numpy as np

path_list_s = [
    "latency_a100/",
    "latency_a40/",
    "latency_a6000/",
    "latency_amd/",
    "latency_P100/",
    "latency_rtx2080/",
    "latency_rtx3080/",
    "latency_v100/",
    "latency_cpu_alldlc/",
    "latency_cpu_a6000/",
    "latency_cpu_meta/",
    "latency_cpu_mlgpu/",
    "latency_cpu_p100/",
]

path_list_m = [
    "latency_a100_m/",
    "latency_a40_m/",
    "latency_a6000_m/",
    "latency_amd_m/",
    "latency_P100_m/",
    "latency_rtx2080_m/",
    "latency_rtx3080_m/",
    "latency_v100_m/",
    "latency_cpu_alldlc_m/",
    "latency_cpu_a6000_m/",
    "latency_cpu_meta_m/",
    "latency_cpu_mlgpu_m/",
    "latency_cpu_p100_m/",
]
path_list_l = [
    "latency_a100_l/",
    "latency_a40_l/",
    "latency_a6000_l/",
    "latency_amd_l/",
    "latency_P100_l/",
    "latency_rtx2080_l/",
    "latency_rtx3080_l/",
    "latency_v100_l/",
    "latency_cpu_alldlc_l/",
    "latency_cpu_a6000_l/",
    "latency_cpu_meta_l/",
    "latency_cpu_mlgpu_l/",
    "latency_cpu_p100_l/",
]


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


import os
import pickle

# check if file exists
base_path = "arch_stats/hwmetric/"
device_type = "gpu"
for path_base in path_list_s:
    if not os.path.exists(path_base):
        print(f"Path {path_base} does not exist")
    else:
        i = 0
        increment = 2500
        cat_list = []
        for i in range(0, 10000, increment):
            path = (
                path_base
                + "efficiency_observations_"
                + str(i)
                + "_"
                + str(i + increment)
                + ".pkl"
            )
            path = path.strip()
            if not os.path.exists(path):
                print(f"Path {path} does not exist")
            else:
                with open(path, "rb") as f:
                    a = pickle.load(f)
                    cat_list.extend(a)
        if "cpu" in path_base:
            print(path_base.split("_"))
            device_name = path_base.split("_")[-2] + "_" + path_base.split("_")[-1][:-1]
            device_type = "cpu"
        else:
            print(path_base.split("_"))
            device_name = path_base.split("_")[-1][:-1]
            device_type = "gpu"
        save_path = base_path + device_name + "_s"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = save_path + "/results.pkl"
        new_stats = process(cat_list, device_type=device_type)
        with open(save_path, "wb") as f:
            pickle.dump(new_stats, f)

base_path = "arch_stats/hwmetric/"
device_type = "gpu"
for path_base in path_list_m:
    if not os.path.exists(path_base):
        print(f"Path {path_base} does not exist")
    else:
        i = 0
        increment = 2500
        cat_list = []
        for i in range(0, 10000, increment):
            path = (
                path_base
                + "efficiency_observations_"
                + str(i)
                + "_"
                + str(i + increment)
                + ".pkl"
            )
            path = path.strip()
            if not os.path.exists(path):
                print(f"Path {path} does not exist")
            else:
                with open(path, "rb") as f:
                    a = pickle.load(f)
                    cat_list.extend(a)
        if "cpu" in path_base:
            print(path_base.split("_"))
            device_name = (
                path_base.split("_")[-3] + "_" + path_base.split("_")[-2]
            )  # [:-1]
            device_type = "cpu"
        else:
            print(path_base.split("_"))
            device_name = path_base.split("_")[-2]  # [:-1]
            device_type = "gpu"
        save_path = base_path + device_name + "_m"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = save_path + "/results.pkl"
        new_stats = process(cat_list, device_type=device_type)
        with open(save_path, "wb") as f:
            pickle.dump(new_stats, f)

base_path = "arch_stats/hwmetric/"
device_type = "gpu"
for path_base in path_list_l:
    if not os.path.exists(path_base):
        print(f"Path {path_base} does not exist")
    else:
        i = 0
        increment = 2500
        cat_list = []
        for i in range(0, 10000, increment):
            path = (
                path_base
                + "efficiency_observations_"
                + str(i)
                + "_"
                + str(i + increment)
                + ".pkl"
            )
            path = path.strip()
            if not os.path.exists(path):
                print(f"Path {path} does not exist")
            else:
                with open(path, "rb") as f:
                    a = pickle.load(f)
                    cat_list.extend(a)
        if "cpu" in path_base:
            print(path_base.split("_"))
            device_name = (
                path_base.split("_")[-3] + "_" + path_base.split("_")[-2]
            )  # [:-1]
            device_type = "cpu"
        else:
            print(path_base.split("_"))
            device_name = path_base.split("_")[-2]  # [:-1]
            device_type = "gpu"
        save_path = base_path + device_name + "_l"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = save_path + "/results.pkl"
        new_stats = process(cat_list, device_type=device_type)
        with open(save_path, "wb") as f:
            pickle.dump(new_stats, f)
