from carbontracker.tracker import CarbonTracker
import torch
import os
import shutil


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
    if use_cpu:
        emissions_cpu = []
        energy_cpu = []
        model = model.cpu()
        input = input.cpu()
        if os.path.exists("log_t4/") and os.path.isdir("log_t4/"):
            shutil.rmtree("log_t4/", ignore_errors=True)
        tracker = CarbonTracker(
            epochs=1, log_dir="log_t4/", components="cpu", verbose=-1
        )
        n = 100
        for epoch in range(100):
            model.eval()
            tracker.epoch_start()
            with torch.no_grad():
                model(input)
            tracker.epoch_end()

        # Optional: Add a stop in case of early termination before all monitor_epochs has
        # been monitored to ensure that actual consumption is reported.

        from carbontracker import parser

        from carbontracker import parser

        tracker.stop()
        logs = parser.parse_all_logs(log_dir="log_t4/")
        # parser.print_aggregate(log_dir="log_t4/")
        first_log = logs[0]
        print("Energy overall:", first_log["pred"]["energy (kWh)"] / n)
        print("C02 overall:", first_log["pred"]["co2eq (g)"] / n)

        mean_co2_cpu = first_log["pred"]["co2eq (g)"] / n
        std_co2_cpu = 0
        mean_energy_cpu = first_log["pred"]["energy (kWh)"] / n
        std_energy_cpu = 0
        print("Energy overall:", first_log["pred"]["energy (kWh)"] / n)
        print("C02 overall:", first_log["pred"]["co2eq (g)"] / n)
    if use_gpu:
        emissions_gpu = []
        energy_gpu = []
        model = model.cuda()
        input = input.cuda()
        if os.path.exists("log_t4/") and os.path.isdir("log_t4/"):
            shutil.rmtree("log_t4/", ignore_errors=True)
        tracker = CarbonTracker(
            epochs=1, log_dir="log_t4/", components="gpu", verbose=-1
        )
        n = 100
        for epoch in range(100):
            model.eval()
            tracker.epoch_start()
            with torch.no_grad():
                model(input)
            tracker.epoch_end()

        # Optional: Add a stop in case of early termination before all monitor_epochs has
        # been monitored to ensure that actual consumption is reported.

        from carbontracker import parser

        from carbontracker import parser

        tracker.stop()
        logs = parser.parse_all_logs(log_dir="log_t4/")
        # parser.print_aggregate(log_dir="log_t4/")
        first_log = logs[0]
        print("Energy overall:", first_log["pred"]["energy (kWh)"] / n)
        print("C02 overall:", first_log["pred"]["co2eq (g)"] / n)

        mean_co2_gpu = first_log["pred"]["co2eq (g)"] / n
        std_co2_gpu = 0
        mean_energy_gpu = first_log["pred"]["energy (kWh)"] / n
        std_energy_gpu = 0
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
        emissions_gpu,
        energy_gpu,
        emissions_cpu,
        energy_cpu,
    )
