import re
import subprocess
import threading
import time

import torch
import wandb
import xmltodict
from colorama import Fore
from colorama import Style


class Tracer(threading.Thread):
    def __init__(self, gpu_num=(0,), profiling_interval=0.1):
        threading.Thread.__init__(
            self,
        )
        self._running = True
        self.power_readings = []
        self.temperature_readings = []
        self.gpu_utils = []
        self.mem_utils = []
        self.counters = 0
        self.profiling_interval = profiling_interval
        self.gpu_num = gpu_num

    def terminate(self):
        self._running = False

    def communicate(self):
        avg_tempr = sum(self.temperature_readings) / (self.counters)
        avg_power = sum(self.power_readings) / (self.counters)
        avg_gpu_utils = sum(self.gpu_utils) / (self.counters)
        avg_mem_utils = sum(self.mem_utils) / (self.counters)
        return (
            round(avg_power, 6),
            round(avg_tempr, 6),
            round(avg_gpu_utils, 6),
            round(avg_mem_utils, 6),
            self.power_readings,
            self.gpu_utils,
            self.mem_utils,
        )

    def run(self):
        while self._running:
            time.sleep(self.profiling_interval)
            self.counters += 1
            results = subprocess.check_output(["nvidia-smi", "-q", "-x"]).decode(
                "utf-8"
            )
            dict_results = xmltodict.parse(results)
            if dict_results["nvidia_smi_log"]["attached_gpus"] == "1":
                single_gpu_info = dict_results["nvidia_smi_log"]["gpu"]
                self.power_readings.append(
                    float(
                        re.findall(
                            r"\d+\.?\d*",
                            single_gpu_info["gpu_power_readings"]["power_draw"],
                        )[0]
                    )
                )
                self.temperature_readings.append(
                    float(
                        re.findall(
                            r"\d+\.?\d*", single_gpu_info["temperature"]["gpu_temp"]
                        )[0]
                    )
                )
                self.gpu_utils.append(
                    float(
                        re.findall(
                            r"\d+\.?\d*", single_gpu_info["utilization"]["gpu_util"]
                        )[0]
                    )
                )
                self.mem_utils.append(
                    float(
                        re.findall(
                            r"\d+\.?\d*", single_gpu_info["utilization"]["memory_util"]
                        )[0]
                    )
                )
            # REMINDS: when running on multiple-gpus, be sure all gpus are activated.
            else:
                # print(dict_results)
                for item in self.gpu_num:
                    self.power_readings.append(
                        float(
                            re.findall(
                                r"\d+\.?\d*",
                                dict_results["nvidia_smi_log"]["gpu"][item][
                                    "gpu_power_readings"
                                ]["power_draw"],
                            )[0]
                        )
                    )
                    self.temperature_readings.append(
                        float(
                            re.findall(
                                r"\d+\.?\d*",
                                dict_results["nvidia_smi_log"]["gpu"][item][
                                    "temperature"
                                ]["gpu_temp"],
                            )[0]
                        )
                    )
                    self.gpu_utils.append(
                        float(
                            re.findall(
                                r"\d+\.?\d*",
                                dict_results["nvidia_smi_log"]["gpu"][item][
                                    "utilization"
                                ]["gpu_util"],
                            )[0]
                        )
                    )
                    self.mem_utils.append(
                        float(
                            re.findall(
                                r"\d+\.?\d*",
                                dict_results["nvidia_smi_log"]["gpu"][item][
                                    "utilization"
                                ]["memory_util"],
                            )[0]
                        )
                    )


"""
Args:
    mode(string): choose mode among supported algorithms, e.g. 'distillation', 'pruning', 'quantization', 'nas'.
    gpu_num(tuple): specify the GPU indexes (multiple GPU supported) that the tracer would need to monitor.
    profiling_interval(float): specify the profile interval, e.g. every 0.1s.
"""


class GPUTracer:
    all_modes = ["distillation", "pruning", "quantization", "nas", "normal"]
    is_enable = True

    def __init__(self, mode, gpu_num=(0,), profiling_interval=0.1, verbose=False):
        if not mode in GPUTracer.all_modes:
            raise ValueError(f"Invalid mode : {mode}")

        self.mode = mode
        self.gpu_num = gpu_num
        self.profiling_interval = profiling_interval
        self.verbose = verbose

    def wrapper(self, *args, **kwargs):

        if not GPUTracer.is_enable:
            return self.func(*args, **kwargs), None

        tracer = Tracer(
            gpu_num=self.gpu_num, profiling_interval=self.profiling_interval
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        tracer.start()
        results = self.func(*args, **kwargs)
        tracer.terminate()
        end.record()
        torch.cuda.synchronize()

        if tracer.counters == 0:
            print("*" * 50)
            print("No tracing info collected, increasing sampling rate if needed.")
            print("*" * 50)
            tracer.join()
            return results, None
        else:
            tracer.join()
            (
                avg_power,
                avg_temperature,
                avg_gpu_utils,
                avg_mem_utils,
                total_power,
                total_gpu_utils,
                total_mem_utils,
            ) = tracer.communicate()
            time_elapse = start.elapsed_time(end) / 1000
            energy_consumption = time_elapse * avg_power / 3600
            # if set verbose=True, it will print the gpu information everytime you call this function.
            if self.verbose:
                print("*" * 50)
                print(
                    f"Time Elapse: {Fore.GREEN}%.2f{Style.RESET_ALL} (in S)"
                    % time_elapse
                )
                print(
                    f"Average Power: {Fore.GREEN}%.2f{Style.RESET_ALL} (in W)"
                    % avg_power
                )
                print(
                    f"Average Temeperature: {Fore.GREEN}%.2f{Style.RESET_ALL} (in C)"
                    % avg_temperature
                )
                print(
                    f"Energy Consumption: {Fore.GREEN}%.2f{Style.RESET_ALL} (in KWh)"
                    % energy_consumption
                )
                print(
                    f"GPU Utils: {Fore.GREEN}%.2f{Style.RESET_ALL} (in percent)"
                    % avg_gpu_utils
                )
                print(
                    f"Mem Utils: {Fore.GREEN}%.2f{Style.RESET_ALL} (in percent)"
                    % avg_mem_utils
                )
                print(
                    "excel compatible output:\n"
                    "{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
                        avg_gpu_utils,
                        avg_mem_utils,
                        energy_consumption,
                        avg_power,
                        avg_temperature,
                        time_elapse,
                    )
                )
                print("*" * 50)

            return results, {
                "Time Elapse": time_elapse,
                "Average Power": avg_power,
                "Average Temperature": avg_temperature,
                "Energy Consumption": energy_consumption,
                "GPU Utils": avg_gpu_utils,
                "Mem Utils": avg_mem_utils,
                "Total Power": total_power,
                "Total GPU Utils": total_gpu_utils,
                "Total Mem Utils": total_mem_utils,
            }

    def __call__(self, func):
        self.func = func
        return self.wrapper

    def enable_trace(self):
        GPUTracer.is_enable = True
        return

    def disable_trace(self):
        GPUTracer.is_enable = False
        return


"""# Example usage:
@GPUTracer(mode='normal', gpu_num=(0,), verbose=True)
@torch.no_grad()
def foo():
    for i in range(10):
        a = torch.rand(1000, 1000).cuda()
        b = torch.rand(1000, 1000).cuda()
        c = torch.mm(a, b)
    #return c

results = foo()
print(results)"""
