from deepspeed.profiling.flops_profiler import FlopsProfiler
from accelerate import Accelerator
import os
from deepspeed.accelerator import get_accelerator
os.environ["DS_ACCELERATOR"]="cpu"

def get_flops_macs_params(model, input):
    #use cpu for profiling
    model = model.cpu()
    input = input.cpu()
    os.environ["DS_ACCELERATOR"]="cpu"
    input = input.cpu()
    model = model.cpu()
    os.environ["DS_ACCELERATOR"]="cpu"
    prof = FlopsProfiler(model)
    os.environ["DS_ACCELERATOR"]="cpu"
    # set deepspeed accelerator to cpu
    prof.start_profile()
    print(os.environ["DS_ACCELERATOR"])
    try:
     model(input)
    except:
        print("Error in getting flops")
    prof.stop_profile()
    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()
    #latest_layer_flops = prof.get_total_duration()
    return flops, macs, params