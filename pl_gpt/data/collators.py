from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch
from torch.nn.utils.rnn import pad_sequence


class DataCollator:

    def __init__(self, src_mask_token_id, trg_mask_token_id):
        self.src_mask_token_id = src_mask_token_id
        self.trg_mask_token_id = trg_mask_token_id

    def __call__(self, id_samples) -> List[List[int]]:
        # tokenize the input text samples

        batch = {}
        src_text_ids = []
        trg_post_text_ids = []
        seq_length = []
        for sample in id_samples:
            src_text_ids.append(torch.LongTensor(sample[:-1]))
            trg_post_text_ids.append(torch.LongTensor(sample[1:]))
            seq_length.append(len(sample) - 1)

        batch["src_len"] = torch.LongTensor(seq_length).contiguous()
        batch["trg_len"] = torch.LongTensor(seq_length).contiguous()

        if torch.all(batch["src_len"] == batch["src_len"][0]):
            batch["src_seq"] = torch.stack(src_text_ids, dim=0).contiguous()
            batch["trg_seq"] = torch.stack(trg_post_text_ids, dim=0).contiguous()
        else:
            batch["src_seq"] = pad_sequence(
                src_text_ids, batch_first=True, padding_value=self.src_mask_token_id
            ).contiguous()
            batch["trg_seq"] = pad_sequence(
                trg_post_text_ids,
                batch_first=True,
                padding_value=self.trg_mask_token_id,
            ).contiguous()

        return batch
