from __future__ import annotations

import json
import os
from pathlib import Path
from pprint import pprint

import torch
import transformers

from whittle.eval.whittle_llms import WhittleLM
from whittle.models.gpt import GPT


def prepare_results(results, save_filepath, print_results=True):
    from lm_eval.utils import make_table

    if print_results:
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

    json_result = json.dumps(results, indent=2, ensure_ascii=False, default=str)
    save_filepath.open("w", encoding="utf-8").write(json_result)


def convert_and_evaluate(
    model: GPT,
    tasks: str | None = None,
    out_dir=None,
    force_conversion: bool = False,
    num_fewshot: int | None = None,
    batch_size: int | str = 1,
    device: str | None = None,
    dtype: str | torch.dtype | None = None,
    limit: float | None = None,
    seed: int = 1234,
    save_filepath: Path | None = None,
    access_token: str | None = None,
    tokenizer: str | transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast = None
) -> None:
    """Evaluate a model with the LM Evaluation Harness.

    Arguments:
        checkpoint_dir: Directory where the `lit_model.pth` and tokenizer files are located.
        out_dir: Directory in which to save the converted checkpoints for evaluation.
            Saves to `checkpoint_dir`/evaluate by default.
        force_conversion: Set to `True` to reconvert the model and override
            an existing model.pth from a previous evaluation call.
        tasks: CSV of task names to evaluate. Example: "hellaswag,truthfulqa_mc2,mmlu"
        num_fewshot: Number of examples in few-shot context.
        batch_size: Batch size configuration as positive integer value (default: 1),
            "auto", in the format 'auto:N', where 'auto:4' recomputes the batch size 4 times.
        device: Device to use for evaluation, for example, "cuda" or "cuda:0".
        limit: Limit on number of examples per task.
        seed: Random seed.
        save_filepath: The file where the results will be saved.
            Saves to `out_dir/results.json` by default.
        access_token: Optional API token to access models with restrictions.
    """
    if tasks is None:
        from lm_eval.tasks import TaskManager

        taskm = TaskManager()
        print("\n".join(taskm.task_index.keys()))
        print(
            "\n\nTo evaluate multiple tasks, you can chain the task names "
            "listed above via a comma-separated list."
            "\nFor example: `--tasks 'hellaswag,truthfulqa_mc2,mmlu'`. "
            "\nTo search for a specific task, use `litgpt evaluate list | grep task_name`."
        )
        return

    pprint(locals())

    if not (isinstance(batch_size, int) and batch_size > 0) and not (
        isinstance(batch_size, str) and batch_size.startswith("auto")
    ):
        raise ValueError(
            "batch_size must be a positive integer, 'auto', or in the format 'auto:N'."
        )

    from lm_eval import evaluator

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = WhittleLM(
        pretrained=model, device=device, batch_size=batch_size, dtype=dtype, tokenizer = tokenizer
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    results = evaluator.simple_evaluate(
        model=model,
        tasks=tasks.split(","),
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_filepath = (
        out_dir / Path("results.json") if save_filepath is None else Path(save_filepath)
    )
    prepare_results(results, save_filepath)
