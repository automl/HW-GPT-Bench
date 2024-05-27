import pickle
import numpy as np

i = 0
increment = 2500
cat_list = []
for i in range(0, 10000, increment):
    path = (
        "/work/dlclarge2/sukthank-hw-llm-bench/HW-Aware-LLM-Bench/latency_cpu_meta/efficiency_observations_"
        + str(i)
        + "_"
        + str(i + increment)
        + ".pkl"
    )
    with open(path, "rb") as f:
        a = pickle.load(f)
        cat_list.extend(a)
a = cat_list

print(len(a))
embedding_dim_768 = []
embedding_dim_384 = []
embedding_dim_192 = []
print(a[0])
for i in a:
    # remove outliers
    energy_gpu = i["mean_energy_cpu"]
    if i["arch"]["sample_embed_dim"] == 768:
        embedding_dim_768.append(np.mean(energy_gpu))
    if i["arch"]["sample_embed_dim"] == 384:
        embedding_dim_384.append(np.mean(energy_gpu))
    if i["arch"]["sample_embed_dim"] == 192:
        embedding_dim_192.append(np.mean(energy_gpu))
layers_12 = []
layers_10 = []
layers_11 = []
for i in a:
    energy_gpu = i["mean_energy_cpu"]
    if i["arch"]["sample_n_layer"] == 12:
        layers_12.append(np.mean(energy_gpu))
    if i["arch"]["sample_n_layer"] == 10:
        layers_10.append(np.mean(energy_gpu))
    if i["arch"]["sample_n_layer"] == 11:
        layers_11.append(np.mean(energy_gpu))


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.grid(linestyle="--")
plt.ecdf(layers_12, label="12", color="r")
plt.ecdf(layers_10, label="10", color="b")
plt.ecdf(layers_11, label="11", color="g")

plt.legend()
plt.xlabel("Energy (Wh)")
plt.ylabel("CDF")
plt.title("CDF of Energy for different number of layers")
plt.savefig("ecdf_plots/energy_cdf_cpu_meta_layers.pdf")
plt.clf()

plt.grid(linestyle="--")
plt.ecdf(embedding_dim_768, label="768", color="r")
plt.ecdf(embedding_dim_384, label="384", color="b")
plt.ecdf(embedding_dim_192, label="192", color="g")

plt.legend()
plt.xlabel("Energy (Wh)")
plt.ylabel("CDF")
plt.title("CDF of Energy for different embedding dims")
plt.savefig("ecdf_plots/energy_cdf_cpu_meta_embed_dim.pdf")
