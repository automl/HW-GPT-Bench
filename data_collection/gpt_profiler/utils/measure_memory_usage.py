import torch


def compute_memory_consumed(
    model: torch.nn.Module,
    input: torch.Tensor,
    n: int = 10,
    use_gpu: bool = True,
    use_cpu: bool = True,
    gpu_dtype: torch.dtype = torch.bfloat16,
):
    if use_gpu:
        memory = []
        model = model.cuda()
        input = input.cuda()
        for i in range(n):
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=gpu_dtype):
                    model(input)
                    # aqppend allocated memory
                    if i > 2:
                        memory.append(float(torch.cuda.memory_allocated()) / 1e6)
                    print(memory)
                    # print(torch.cuda.mem_get_info())
        # return the average memory consumed
        mean_memory = torch.tensor(memory).mean().item()
        std_memory = torch.tensor(memory).std().item()

        return mean_memory, std_memory
