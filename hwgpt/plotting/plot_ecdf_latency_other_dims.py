import pickle
import numpy as np

i = 0
increment = 2500
cat_list = []
for i in range(0, 10000, increment):
    path = (
        "/work/dlclarge2/sukthank-hw-llm-bench/HW-Aware-LLM-Bench/latency_h100/efficiency_observations_"
        + str(i)
        + "_"
        + str(i + increment)
        + ".pkl"
    )
    with open(path, "rb") as f:
        a = pickle.load(f)
        cat_list.extend(a)
a = cat_list

max_mlp_ratio_sum = 4 * 12
min_mlp_ratio_sum = 2 * 12
equispaced_mlp_ratios = np.linspace(min_mlp_ratio_sum, max_mlp_ratio_sum, 9)[
    2:7
]  # [0:10]
binned_mlp_ratios = {}
for i in range(len(equispaced_mlp_ratios)):
    binned_mlp_ratios[str(equispaced_mlp_ratios[i])] = []


for i in a:
    if "cpu" in path:
        latencies = i["times_profiler_cpu"]
        units = i["unit_cpu"]
    else:
        latencies = i["times_profiler_gpu"]
        units = i["unit_gpu"]
    latencies_scaled = []
    for j in range(len(latencies)):
        if units[j] == "ms":
            latencies_scaled.append(latencies[j])
        elif units[j] == "s":
            latencies_scaled.append(latencies[j] * 1000)
    sum_mlp = np.sum(i["arch"]["sample_mlp_ratio"])
    for j in range(len(equispaced_mlp_ratios) - 1):
        if (
            sum_mlp >= equispaced_mlp_ratios[j]
            and sum_mlp < equispaced_mlp_ratios[j + 1]
        ):
            binned_mlp_ratios[str(equispaced_mlp_ratios[j])].append(
                np.mean(latencies_scaled)
            )


for i in range(len(equispaced_mlp_ratios)):
    print(len(binned_mlp_ratios[str(equispaced_mlp_ratios[i])]))
    # delete if len is 0
    if len(binned_mlp_ratios[str(equispaced_mlp_ratios[i])]) < 10:
        del binned_mlp_ratios[str(equispaced_mlp_ratios[i])]

max_num_heads = 12 * 12
min_num_heads = 4 * 12
equispaced_num_heads = np.linspace(min_num_heads, max_num_heads, 9)[2:7]  # [0:10]
binned_num_heads = {}
for i in range(len(equispaced_num_heads)):
    binned_num_heads[str(equispaced_num_heads[i])] = []


for i in a:
    if "cpu" in path:
        latencies = i["times_profiler_cpu"]
        units = i["unit_cpu"]
    else:
        latencies = i["times_profiler_gpu"]
        units = i["unit_gpu"]
    latencies_scaled = []
    for j in range(len(latencies)):
        if units[j] == "ms":
            latencies_scaled.append(latencies[j])
        elif units[j] == "s":
            latencies_scaled.append(latencies[j] * 1000)
    sum_num_heads = np.sum(i["arch"]["sample_n_head"])
    for j in range(len(equispaced_num_heads) - 1):
        if (
            sum_num_heads >= equispaced_num_heads[j]
            and sum_num_heads < equispaced_num_heads[j + 1]
        ):
            binned_num_heads[str(equispaced_num_heads[j])].append(
                np.mean(latencies_scaled)
            )


for i in range(len(equispaced_num_heads)):
    print(len(binned_num_heads[str(equispaced_num_heads[i])]))
    # delete if len is 0
    if len(binned_num_heads[str(equispaced_num_heads[i])]) < 10:
        del binned_num_heads[str(equispaced_num_heads[i])]

bias_true = []
bias_false = []
for i in a:
    if "cpu" in path:
        latencies = i["times_profiler_cpu"]
        units = i["unit_cpu"]
    else:
        latencies = i["times_profiler_gpu"]
        units = i["unit_gpu"]
    latencies_scaled = []
    for j in range(len(latencies)):
        if units[j] == "ms":
            latencies_scaled.append(latencies[j])
        elif units[j] == "s":
            latencies_scaled.append(latencies[j] * 1000)
    if i["arch"]["sample_bias"] == True:
        bias_true.append(np.mean(latencies_scaled))
    if i["arch"]["sample_bias"] == False:
        bias_false.append(np.mean(latencies_scaled))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.grid(linestyle="--")
for key in binned_mlp_ratios:
    plt.ecdf(binned_mlp_ratios[key], label=key)


plt.legend()
plt.xlabel("Latency (ms)")
plt.ylabel("CDF")
plt.title("CDF of Latency for different MLP ratios")
plt.savefig("ecdf_plots/latency_cdf_h100_mlp_ratio.pdf")
plt.clf()

plt.grid(linestyle="--")
for key in binned_num_heads:
    plt.ecdf(binned_num_heads[key], label=key)
plt.legend()
plt.xlabel("Latency (ms)")
plt.ylabel("CDF")
plt.title("CDF of Latency for different number of heads")
plt.savefig("ecdf_plots/latency_cdf_h100_num_heads.pdf")
plt.clf()

plt.grid(linestyle="--")
plt.ecdf(bias_true, label="True", color="r")
plt.ecdf(bias_false, label="False", color="b")
plt.legend()
plt.xlabel("Latency (ms)")
plt.ylabel("CDF")
plt.title("CDF of Latency for different biases")
plt.savefig("ecdf_plots/latency_cdf_h100_bias.pdf")
plt.clf()
