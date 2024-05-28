from codecarbon import OfflineEmissionsTracker
import numpy as np
import torch


def compute_carbon_emissions(
    model: torch.nn.Module,
    input: torch.Tensor,
    n: int = 10,
    use_gpu: bool = True,
    use_cpu: bool = True,
    gpu_dtype: torch.dtype = torch.bfloat16,
):

    mean_co2_cpu = None
    std_co2_cpu = None
    mean_energy_cpu = None
    std_energy_cpu = None
    mean_co2_gpu = None
    std_co2_gpu = None
    mean_energy_gpu = None
    std_energy_gpu = None
    if use_cpu:
        emissions_cpu = []
        energy_cpu = []
        model = model.cpu()
        input = input.cpu()
        for i in range(n):
            emissions_tracker = OfflineEmissionsTracker(
                log_level="error", country_iso_code="DEU"
            )
            emissions_tracker.start()
            with torch.no_grad():
                model(input)
            emissions = emissions_tracker.stop()
            if emissions is None:
                emissions = 0
            emissions_cpu.append(emissions)
            energy_cpu.append(emissions_tracker._total_energy.kWh * 1000)

        mean_co2_cpu = np.mean(emissions_cpu) * 1_000
        std_co2_cpu = np.std(emissions_cpu) * 1_000
        mean_energy_cpu = np.mean(energy_cpu)
        std_energy_cpu = np.std(energy_cpu)
    if use_gpu:
        emissions_gpu = []
        energy_gpu = []
        model = model.cuda()
        input = input.cuda()
        for i in range(n):
            emissions_tracker = OfflineEmissionsTracker(
                log_level="error", country_iso_code="DEU"
            )
            emissions_tracker.start()
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=gpu_dtype):
                    model(input)
            emissions = emissions_tracker.stop()
            if emissions is None:
                emissions = 0
            emissions_gpu.append(emissions)
            energy_gpu.append(emissions_tracker._total_energy.kWh * 1000)
            # print(emissions_tracker._total_energy.kWh)

        mean_co2_gpu = np.mean(emissions_gpu) * 1_000
        std_co2_gpu = np.std(emissions_gpu) * 1_000
        mean_energy_gpu = np.mean(energy_gpu)
        std_energy_gpu = np.std(energy_gpu)
    unit_co2 = "kgCO2"
    unit_energy = "Wh"
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
    )
