import pickle
import numpy as np
with open("/work/dlclarge2/sukthank-hw-llm-bench/HW-Aware-LLM-Bench/predictors/hwmetric/conformal_predictions_rtx2080.pkl","rb") as f:
    a = pickle.load(f)

embedding_dim_768_l = []
embedding_dim_384_l = []
embedding_dim_192_l = []
embedding_dim_768_h = []
embedding_dim_384_h = []
embedding_dim_192_h = []
print(a[0])
print(len(a))
for i in a:
    arch = i[0]
    latencies = i[1]
    if arch["sample_embed_dim"] == 768:
          embedding_dim_768_l.append(latencies[0])    
          embedding_dim_768_h.append(latencies[-1])
    if arch["sample_embed_dim"] == 384:
          embedding_dim_384_l.append(latencies[0])    
          embedding_dim_384_h.append(latencies[-1])
    if arch["sample_embed_dim"] == 192:
          embedding_dim_192_l.append(latencies[0])    
          embedding_dim_192_h.append(latencies[-1])
layers_12_l = []
layers_10_l = []
layers_11_l = []
layers_12_h = []
layers_10_h = []
layers_11_h = []
for i in a:
    arch = i[0]
    latencies = i[1]
    if arch["sample_n_layer"] == 12:
              layers_12_l.append(latencies[0])    
              layers_12_h.append(latencies[-1])
    if arch["sample_n_layer"] == 10:
              layers_10_l.append(latencies[0])    
              layers_10_h.append(latencies[-1])
    if arch["sample_n_layer"] == 11:
              layers_11_l.append(latencies[0])    
              layers_11_h.append(latencies[-1])


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.grid(linestyle='--')
plt.ecdf(layers_12_l, label="12_l",color='r')
plt.ecdf(layers_10_l, label="10_l",color='b')
plt.ecdf(layers_11_l, label="11_l",color='g')
plt.ecdf(layers_12_h, label="12_h",color='r')
plt.ecdf(layers_10_h, label="10_h",color='b')
plt.ecdf(layers_11_h, label="11_h",color='g')


plt.legend()
plt.xlabel("Latency (ms)")
plt.ylabel("CDF")
plt.title("CDF of Latency for different layers: a140")
plt.savefig("ecdf_plots/latency_cdf_rtx2080_layers_conformal.pdf")
plt.clf()

plt.grid(linestyle='--')
plt.ecdf(embedding_dim_768_l, label="768_l",color='r')
plt.ecdf(embedding_dim_384_l, label="384_l",color='b')
plt.ecdf(embedding_dim_192_l, label="192_l",color='g')
plt.ecdf(embedding_dim_768_h, label="768_h",color='r')
plt.ecdf(embedding_dim_384_h, label="384_h",color='b')
plt.ecdf(embedding_dim_192_h, label="192_h",color='g')
print(embedding_dim_768_l[0])
print(embedding_dim_768_h[0])
plt.legend()
plt.xlabel("Latency (ms)")
plt.ylabel("CDF")
plt.title("CDF of Latency for different embedding dims")
plt.savefig("ecdf_plots/latency_cdf_rtx2080_embed_dim_conformal.pdf")