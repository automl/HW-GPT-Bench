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

embedding_dim_768 = []
embedding_dim_384 = []
embedding_dim_192 = []
print(a[0])
print(len(a))
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
    if i["arch"]["sample_embed_dim"] == 768:
        embedding_dim_768.append(np.mean(latencies_scaled))
    if i["arch"]["sample_embed_dim"] == 384:
        embedding_dim_384.append(np.mean(latencies_scaled))
    if i["arch"]["sample_embed_dim"] == 192:
        embedding_dim_192.append(np.mean(latencies_scaled))
layers_12 = []
layers_10 = []
layers_11 = []
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
    if i["arch"]["sample_n_layer"] == 12:
        layers_12.append(np.mean(latencies_scaled))
    if i["arch"]["sample_n_layer"] == 10:
        layers_10.append(np.mean(latencies_scaled))
    if i["arch"]["sample_n_layer"] == 11:
        layers_11.append(np.mean(latencies_scaled))


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.grid(linestyle="--")
plt.ecdf(layers_12, label="12", color="r")
plt.ecdf(layers_10, label="10", color="b")
plt.ecdf(layers_11, label="11", color="g")

plt.legend()
plt.xlabel("Latency (ms)")
plt.ylabel("CDF")
plt.title("CDF of Latency for different layers: h100")
plt.savefig("ecdf_plots/latency_cdf_h100_layers.pdf")
plt.clf()

plt.grid(linestyle="--")
plt.ecdf(embedding_dim_768, label="768", color="r")
plt.ecdf(embedding_dim_384, label="384", color="b")
plt.ecdf(embedding_dim_192, label="192", color="g")

plt.legend()
plt.xlabel("Latency (ms)")
plt.ylabel("CDF")
plt.title("CDF of Latency for different embedding dims")
plt.savefig("ecdf_plots/latency_cdf_h100_embed_dim.pdf")
