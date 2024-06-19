# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import re
import logging

from hwgpt.model.gpt.model import GPT
from hwgpt.model.gpt.config import Config

from lightning_utilities.core.imports import RequirementCache

# Suppress excessive warnings, see https://github.com/pytorch/pytorch/issues/111632
pattern = re.compile(".*Profiler function .* will be ignored")
logging.getLogger("torch._dynamo.variables.torch").addFilter(
    lambda record: not pattern.search(record.getMessage())
)

# Avoid printing state-dict profiling output at the WARNING level when saving a checkpoint
logging.getLogger("torch.distributed.fsdp._optim_utils").disabled = True
logging.getLogger("torch.distributed.fsdp._debug_utils").disabled = True

__all__ = ["GPT", "Config", "Tokenizer"]
