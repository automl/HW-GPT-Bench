from codecarbon import OfflineEmissionsTracker
import numpy as np
import torch
def compute_carbon_emissions(model, input, n = 10, use_gpu=True, use_cpu=True, gpu_dtype=torch.bfloat16):
    emissions_cpu = []
    energy_cpu = []
    model = model.cpu()
    input = input.cpu()
    while len(energy_cpu) < n+2:
      emissions_tracker = OfflineEmissionsTracker(log_level="error",country_iso_code="DEU")
      emissions_tracker.start()
      with torch.no_grad():
        with torch.amp.autocast(device_type="cpu", dtype=torch.float32):
            model(input)
      emissions = emissions_tracker.stop()
      print(emissions_tracker._total_energy.kWh)
      if emissions_tracker._total_energy.kWh != 0:
        emissions_cpu.append(emissions)
        energy_cpu.append(emissions_tracker._total_energy.kWh*1000)
    if use_gpu:
      emissions_gpu = []
      energy_gpu = []
      model = model.cuda()
      input = input.cuda()
      while len(energy_gpu) < n+2:
        emissions_tracker = OfflineEmissionsTracker(log_level="error",country_iso_code="DEU")
        emissions_tracker.start()
        with torch.no_grad():
          with torch.amp.autocast(device_type="cuda", dtype=gpu_dtype):
              model(input)
        emissions = emissions_tracker.stop()
        print(emissions_tracker._total_gpu_energy.kWh)
        if emissions_tracker._total_gpu_energy.kWh != 0:
          #emissions_gpu.append(emissions)
          energy_gpu.append(emissions_tracker._total_gpu_energy.kWh*1000)
        #print(emissions_tracker._total_energy.kWh)

      mean_co2_gpu = np.mean(emissions_gpu)* 1_000
      std_co2_gpu = np.std(emissions_gpu)*1_000
      mean_energy_gpu = np.mean(energy_gpu[2:])
      std_energy_gpu = np.std(energy_gpu[2:])
    unit_co2 = "kgCO2"
    unit_energy = "Wh"
    return mean_co2_cpu, std_co2_cpu, mean_co2_gpu, std_co2_gpu, unit_co2, mean_energy_cpu, std_energy_cpu, mean_energy_gpu, std_energy_gpu, unit_energy, emissions_gpu, energy_gpu, emissions_cpu, energy_cpu



