import math
import torch

import math
import torch


def get_learning_rate_schedule(scheduler_config):
    def lr_lambda(current_step: int):
        if current_step < scheduler_config.num_warmup_steps:
            return float(current_step) / float(
                max(1, scheduler_config.num_warmup_steps)
            )
        elif scheduler_config.schedule == "linear":
            return scheduler_config.decay_factor + (
                1 - scheduler_config.decay_factor
            ) * max(
                0.0,
                float(
                    scheduler_config.num_training_steps
                    - scheduler_config.num_warmup_steps
                    - current_step
                )
                / float(
                    max(
                        1,
                        scheduler_config.num_training_steps
                        - scheduler_config.num_warmup_steps,
                    )
                ),
            )
        elif scheduler_config.schedule == "cosine":
            return scheduler_config.decay_factor + (
                1 - scheduler_config.decay_factor
            ) * max(
                0.0,
                (
                    1
                    + math.cos(
                        math.pi
                        * (current_step - scheduler_config.num_warmup_steps)
                        / float(
                            max(
                                1,
                                scheduler_config.num_training_steps
                                - scheduler_config.num_warmup_steps,
                            )
                        )
                    )
                )
                / 2,
            )
        elif scheduler_config.schedule == "const":
            return 1.0

    return lr_lambda


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from types import SimpleNamespace

    def visualize_lr(optimizer, scheduler, max_step):
        lrs = []
        for step in range(1, max_step):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])
        plt.plot(range(1, max_step), lrs)
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning rate schedule")
        plt.show()

    model = torch.nn.Linear(4, 3)
    lr_max = 0.005

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr_max)

    scheduler_config = SimpleNamespace(
        **{
            "num_training_steps": 10000,
            "num_warmup_steps": 1000,
            "decay_factor": 0.8,
            "schedule": "cosine",
        }
    )

    lr_lambda = get_learning_rate_schedule(scheduler_config)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=-1
    )

    visualize_lr(optimizer, lr_scheduler, scheduler_config.num_training_steps)
