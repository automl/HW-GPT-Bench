import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats_scipy


def calculate_correlation_matrix(all_dict):
    corr_mat = np.zeros((len(all_dict), len(all_dict)))
    keys = list(all_dict.keys())
    for i in range(len(keys)):
        for j in range(len(keys)):
            corr, _ = stats_scipy.kendalltau(all_dict[keys[i]], all_dict[keys[j]])
            corr_mat[i, j] = round(corr, 2)
    return corr_mat, keys


def plot_correlation_matrix(corr_mat, keys, filename):
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_mat, annot=True, xticklabels=keys, yticklabels=keys)
    plt.savefig(filename)


def main():
    hw_agnostic = (
        "params",
        "flops",
        "float16_memory",
        "bfloat16_memory",
        "perplexity",
        "accuracy",
    )
    metrics = (
        "latencies",
        "energies",
        "float16_memory",
        "bfloat16_memory",
        "flops",
        "params",
    )
    metric_map = ("lat", "en", "f16_mem", "bf16_mem", "flops", "params")
    search_space_choices = ("m", "s", "l")
    devices = (
        "a100",
        "a40",
        "h100",
        "rtx2080",
        "rtx3080",
        "a6000",
        "v100",
        "P100",
        "cpu_xeon_silver",
        "cpu_xeon_gold",
        "cpu_amd_7502",
        "cpu_amd_7513",
        "cpu_amd_7452",
    )
    for ss in search_space_choices:
        base_path = "data_collection/gpt_datasets/gpt_" + str(ss) + "/" + "stats.pkl"
        corr_mat_lists = {}
        corr_mat_lists["1/perplexity"] = []
        corr_mat_lists["accuracy"] = []
        with open(base_path, "rb") as f:
            arch_stats = pickle.load(f)
        for arch in arch_stats:
            arch_full = arch_stats[arch]
            corr_mat_lists["1/perplexity"].append(1 / arch_full["perplexity"])
            corr_mat_lists["accuracy"].append(arch_full["accuracy"])
        for i, metric in enumerate(metrics):
            if metric in hw_agnostic:
                corr_mat_lists[metric_map[i]] = []
                for arch in arch_stats:
                    # print(arch_stats[arch]["flops"])
                    if isinstance(arch_stats[arch][metric], list):
                        corr_mat_lists[metric_map[i]].append(
                            np.median(arch_stats[arch][metric])
                        )
                    else:
                        corr_mat_lists[metric_map[i]].append(arch_stats[arch][metric])
                continue
            for device in devices:
                corr_mat_lists[device + "_" + metric_map[i]] = []
                for arch in arch_stats:
                    if isinstance(arch_stats[arch][device][metric], list):
                        corr_mat_lists[device + "_" + metric_map[i]].append(
                            np.median(arch_stats[arch][device][metric])
                        )
                    else:
                        corr_mat_lists[device + "_" + metric_map[i]].append(
                            arch_stats[arch][device][metric]
                        )

        cpu_dict = {}
        for key in corr_mat_lists:
            if ("cpu" in key) or (key in hw_agnostic):
                cpu_dict[key] = corr_mat_lists[key]
        corr_mat_cpu, keys_cpu = calculate_correlation_matrix(cpu_dict)
        # update keys
        plot_correlation_matrix(
            corr_mat_cpu, keys_cpu, "corr_matrix_cpu_" + ss + ".pdf"
        )

        # Save correlation matrix for GPU data only
        keys = list(corr_mat_lists.keys())
        gpu_keys = [key for key in keys if "cpu" not in key]
        gpu_dict = {key: corr_mat_lists[key] for key in gpu_keys}
        corr_mat_gpu, keys_gpu = calculate_correlation_matrix(gpu_dict)
        plot_correlation_matrix(
            corr_mat_gpu, keys_gpu, "corr_matrix_gpu_" + ss + ".pdf"
        )

        # Save correlation matrix for all
        keys = list(corr_mat_lists.keys())
        corr_mat, keys = calculate_correlation_matrix(corr_mat_lists)
        plot_correlation_matrix(corr_mat, keys, "corr_matrix_all_" + ss + ".pdf")


if __name__ == "__main__":
    main()
