import torch
from analysis.test_energy_profiler import GPUTracer
import numpy as np


@GPUTracer(mode="normal", gpu_num=(0,), profiling_interval=0.00001, verbose=False)
@torch.no_grad()
def profile_metrics(dtype, inp, model):
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        model(inp)


def compute_carbon_emissions(
    model, input, n=10, use_gpu=True, use_cpu=True, gpu_dtype=torch.bfloat16
):

    mean_co2_cpu = None
    std_co2_cpu = None
    mean_energy_cpu = None
    std_energy_cpu = None
    mean_co2_gpu = None
    std_co2_gpu = None
    mean_energy_gpu = None
    std_energy_gpu = None
    energy_cpu = []
    energy_gpu = []
    emissions_cpu = []
    emissions_gpu = []
    emissions_gpu = []
    energy_gpu = []
    model = model.cuda()
    model.eval()
    input = input.cuda()
    while len(emissions_gpu) < 50:
        results = profile_metrics(dtype=gpu_dtype, inp=input, model=model)
        # print(results[1])
        if results[1] != None:
            emissions_gpu.append(results[1]["Average Power"])
            energy_gpu.append(results[1]["Energy Consumption"])
    mean_co2_gpu = np.mean(emissions_gpu)
    std_co2_gpu = np.std(emissions_gpu)
    mean_energy_gpu = np.mean(energy_gpu)
    std_energy_gpu = np.std(energy_gpu)
    unit_co2 = "W"
    unit_energy = "KWh"
    return (
        mean_co2_cpu,
        std_co2_cpu,
        mean_co2_gpu,
        std_co2_gpu,
        unit_co2,
        mean_energy_cpu,
        std_energy_cpu,
        mean_energy_gpu,
        std_energy_gpu,
        unit_energy,
        emissions_gpu,
        energy_gpu,
        emissions_cpu,
        energy_cpu,
    )
