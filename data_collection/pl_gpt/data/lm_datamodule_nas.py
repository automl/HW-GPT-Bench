"""
DataModule for working with 1 arrow file for train/val
"""

import os
import pathlib
import logging

import datasets
import numpy as np
import multiprocessing
import pytorch_lightning as pl
import pyarrow as pa
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast

from data_collection.pl_gpt.data.detokenizer import wikitext_detokenize
from data_collection.pl_gpt.data.collators import DataCollator
import os

os.environ["HF_DATASETS_CACHE"] =  "/path/to/datasets/cache"
os.environ["HF_HOME"] = "/path/to/model/cache"
IGNORE_INDEX = -100


class IndexDataset(torch.utils.data.Dataset):
    """
    Wrapper class to hold arrow file dataset indices
    """

    def __init__(self, dataset_indices):
        self.dataset_indices = dataset_indices

    def __getitem__(self, index):
        return self.dataset_indices[index]

    def __len__(self):
        return len(self.dataset_indices)


class PlArrowFileModule(pl.LightningDataModule):
    """
    Datamodule to perform pretraining
    based on 1 train arrow file, 1 val arrow file
    Assumes that pre-processed indices exist
    """

    def __init__(
        self,
        dataset_name,
        num_cpu_worker,
        num_gpu_worker,
        max_sample_len,
        seed,
        batch_size,
        data_dir,
        cache_dir,
        val_ratio,
        val_split_seed,
    ):
        super().__init__()

        self.num_gpu_worker = num_gpu_worker

        if num_cpu_worker is None:
            num_cpu_worker = os.cpu_count()
        self.num_cpu_worker = num_cpu_worker

        self.resume_index = None  # TODO not implemented yet
        self.dataset_name = dataset_name

        self.data_dir = pathlib.Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.max_sample_len = max_sample_len
        self.seed = seed

        self.val_sets_name = [dataset_name]

        self.logger = logging.getLogger(__name__)

        self.splits = ["validationnas"]  # 'train', 'validation']
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        if self.dataset_name == "openwebtext":
            self.val_cfg_str = f"val-{str(val_ratio).split('.')[1]}-{val_split_seed}_"
        else:
            self.val_cfg_str = f""
        self.ignore_index = IGNORE_INDEX

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=cache_dir)
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        self.seq_vocab_size = int(np.ceil(len(self.tokenizer) / 128)) * 128
        self.trg_vocab_size = int(np.ceil(len(self.tokenizer) / 128)) * 128
        self.vocab_size = len(self.tokenizer)

        self.global_rank = 0

        self.collator = DataCollator(
            src_mask_token_id=self.tokenizer.pad_token_id,
            trg_mask_token_id=self.ignore_index,
        )

        self.prepare_data()  # pre-call to avoid initialized distributed pytroch for distributed pre-process

    def prepare_data(self):
        if not self._exist_preprocessed_data():
            self._preprocess_data()

    def _exist_preprocessed_data(self):
        all_files_exist = True
        print(self.splits)
        for split in self.splits:
            base_file = f"{self.dataset_name}_{split}_{self.val_cfg_str}{self.max_sample_len +1}_{self.num_gpu_worker}"
            for worker_id in range(self.num_gpu_worker):
                file_name = base_file + f"_{worker_id}.arrow"
                file_exist = os.path.exists(self.data_dir / file_name)
                all_files_exist &= file_exist
                if not file_exist:
                    self.logger.info(
                        f"Checked preprocessed data: {(self.data_dir / file_name).as_posix()} does not exist."
                    )
        if all_files_exist:
            self.logger.info("Checked preprocessed data: All file exist.")
        return all_files_exist

    def _preprocess_data(self):

        max_sample_len_plus = (
            self.max_sample_len + 1
        )  # Because of source target shift in decoder/teacher-forcing training

        if self.dataset_name == "openwebtext":
            all_samples = load_dataset(
                "openwebtext", cache_dir=self.cache_dir.absolute().as_posix()
            )
        elif self.dataset_name == "tinystories":
            print(self.data_dir.absolute().as_posix())
            print(self.cache_dir.absolute().as_posix())
            all_samples = load_dataset(
                "roneneldan/TinyStories", cache_dir=self.cache_dir.absolute().as_posix()
            )
        elif self.dataset_name == "wikitext":
            all_samples = load_dataset(
                "wikitext",
                "wikitext-103-v1",
                cache_dir=self.cache_dir.absolute().as_posix(),
            )
            all_samples = all_samples.map(
                lambda example: {"text": wikitext_detokenize(example["text"])},
                num_proc=max(self.num_cpu_worker, 1),
                desc="Running detokenizer on dataset",
            )
        else:
            raise UserWarning(f"dataset name unknown: {self.dataset_name}")
        print(all_samples.keys())
        if "validation" not in all_samples:
            all_samples = all_samples["train"].train_test_split(
                test_size=self.val_ratio,
                seed=self.val_split_seed,
                shuffle=True,  # Otherwise test will be at the end of the dataset
            )
            all_samples["validation"] = all_samples["test"]

        if "validationnas" not in all_samples:
            val_nas_ratio = 0.1
            all_samples_nas = all_samples["train"].train_test_split(
                test_size=val_nas_ratio,
                seed=self.val_split_seed,
                shuffle=True,  # Otherwise test will be at the end of the dataset
            )
            all_samples["validationnas"] = all_samples_nas["test"]
            # self.splits.append('validationnas')
        for k in all_samples.keys():
            print(k)
            print(len(all_samples[k]))
        print(self.splits)
        for split in self.splits:

            samples = all_samples[split]

            base_file = f"{self.dataset_name}_{split}_{self.val_cfg_str}{max_sample_len_plus}_{self.num_gpu_worker}"
            file_name = (self.data_dir / base_file).as_posix()

            numb_samples = len(samples)
            avg_length = np.mean([len(s["text"]) for s in samples])
            self.logger.info(
                f"split {split} load {numb_samples} data samples with avg length of {avg_length}"
            )

            if multiprocessing.cpu_count() < 16:
                raise UserWarning(
                    f"preprocess requires at least {self.num_gpu_worker * 2} cpus"
                )

            worker_world_size = (
                multiprocessing.cpu_count() // self.num_gpu_worker - 1
            ) * self.num_gpu_worker
            assert worker_world_size % self.num_gpu_worker == 0

            index_list = list(range(numb_samples))
            index_step = numb_samples // worker_world_size

            pa_type = pa.list_(pa.uint16() if self.vocab_size < 65535 else pa.uint32())
            batch = pa.RecordBatch.from_arrays(
                [pa.array([list(range(max_sample_len_plus))], type=pa_type)],
                names=["text"],
            )

            self.logger.info(f"split {split} start parallel {worker_world_size} worker")

            return_queues = []
            for _ in range(self.num_gpu_worker):
                return_queues.append(multiprocessing.Queue(maxsize=100))

            memory_manager_list = []

            for worker_idx in range(self.num_gpu_worker):
                self.logger.info(f"start queue2file_writer process {worker_idx}")
                memory_manager = multiprocessing.Process(
                    target=self._queue2file_writer,
                    args=(
                        file_name,
                        batch,
                        worker_idx,
                        worker_world_size,
                        return_queues,
                    ),
                )
                memory_manager.daemon = True
                memory_manager.start()
                memory_manager_list.append(memory_manager)

            for worker_idx in range(worker_world_size):
                self.logger.info(f"start preprocess_samples2queue process {worker_idx}")
                indexes = index_list[
                    worker_idx * index_step : (1 + worker_idx) * index_step
                ]
                worker = worker_idx % self.num_gpu_worker
                memory_manager = multiprocessing.Process(
                    target=self._preprocess_samples2queue,
                    args=(
                        self.tokenizer,
                        samples,
                        indexes,
                        max_sample_len_plus,
                        worker,
                        return_queues,
                        pa_type,
                    ),
                )
                memory_manager.daemon = True
                memory_manager.start()
                memory_manager_list.append(memory_manager)

            for memory_manager in memory_manager_list:
                memory_manager.join()

            self.logger.info(f"split {split} preprocess done")

    @staticmethod
    def _queue2file_writer(file_name, batch, worker, worker_world_size, return_queues):

        total_samples = 0
        with pa.OSFile(f"{file_name}_{worker}.arrow", "wb") as sink:
            with pa.ipc.new_file(sink, batch.schema) as writer:
                end_count = 0
                while end_count < worker_world_size // len(return_queues):
                    batch = return_queues[worker].get()
                    if batch == "END":
                        end_count += 1
                    else:
                        writer.write_batch(batch)
                        total_samples += 1
        return_queues[worker].close()
        print(
            f"queue2file_writer {worker} wrote {total_samples} in {file_name}_{worker}.arrow"
        )

    @staticmethod
    def _preprocess_samples2queue(
        tokenizer, samples, indexes, max_sample_len_plus, worker, return_queues, pa_type
    ):

        count_writes = 0
        idx_sample = 0
        tmp_sample = []
        numb_samples = len(indexes)

        while idx_sample < numb_samples:
            if len(tmp_sample) < max_sample_len_plus:
                raw_text = samples[indexes[idx_sample]]["text"]
                tmp_sample += tokenizer(raw_text)["input_ids"] + [
                    tokenizer.eos_token_id
                ]
                idx_sample += 1
            else:
                arr, tmp_sample = (
                    pa.array([tmp_sample[:max_sample_len_plus]], type=pa_type),
                    tmp_sample[max_sample_len_plus:],
                )
                batch = pa.RecordBatch.from_arrays([arr], names=["text"])
                return_queues[worker].put(batch)
                count_writes += 1

        return_queues[worker].put("END")
        print(
            f"preprocess_samples2queue {worker} done: processed {idx_sample} data samples, created {count_writes} training samples"
        )

    def setup(self, stage: str):

        if torch.distributed.is_initialized():
            self.global_rank = torch.distributed.get_rank()

        self.rng = np.random.RandomState(self.seed + self.global_rank)

        self.logger.info("Create memory map\n")
        # train_file_name = f"{self.dataset_name}_train_{self.val_cfg_str}{self.max_sample_len +1}_{self.num_gpu_worker}_{self.global_rank}.arrow"
        # mmap = pa.memory_map( (self.data_dir / train_file_name).as_posix() )
        # self.logger.info("MMAP Read ALL")
        # self._train_dataset = pa.ipc.open_file(mmap)

        # valid_file_name = f"{self.dataset_name}_validation_{self.val_cfg_str}{self.max_sample_len +1}_{self.num_gpu_worker}_{self.global_rank}.arrow"
        # valid_mmap = pa.memory_map((self.data_dir / valid_file_name).as_posix() )
        # self._valid_dataset = pa.ipc.open_file(valid_mmap)

        valid_nas_file_name = f"{self.dataset_name}_validationnas_{self.val_cfg_str}{self.max_sample_len +1}_{self.num_gpu_worker}_{self.global_rank}.arrow"
        valid_nas_mmap = pa.memory_map((self.data_dir / valid_nas_file_name).as_posix())
        self._valid_nas_dataset = pa.ipc.open_file(valid_nas_mmap)

    def train_dataloader(self):
        """This will be run every epoch."""

        if torch.distributed.is_initialized():
            global_rank = torch.distributed.get_rank()

        world_size = torch.cuda.device_count()

        local_rank = global_rank % world_size  # TODO CONFIGRUABEL

        train_set_size = self._train_dataset.num_record_batches
        train_indexes = list(range(train_set_size))
        train_indexes = self.rng.permutation(train_indexes)

        # min_num_samples = torch.LongTensor([train_set_size]).to(local_rank)
        min_num_samples = torch.tensor(train_set_size, device=f"cuda:{local_rank}")
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                min_num_samples, op=torch.distributed.ReduceOp.MIN
            )
        min_num_samples = min_num_samples.item()
        train_indexes = train_indexes[:min_num_samples]

        self.logger.info(
            f"### load train set with size {min_num_samples} from {train_set_size} samples on rank {self.global_rank}"
        )

        # shuffle the indices for every epoch other than 0.
        # the loaded indices are already shuffled
        if self.trainer.current_epoch > 0:
            seed = self.seed + self.trainer.current_epoch + self.global_rank
            tmp_rng = np.random.default_rng(seed)
            train_indexes = tmp_rng.permutation(train_indexes)

        if self.resume_index is not None:
            train_indexes = train_indexes[self.resume_index :]
            self.resume_index = None  # reset to avoid next-epoch issues

        train_index_dataset = IndexDataset(train_indexes)

        def train_pl_collate_fn(indices):
            raw_samples = [
                self._train_dataset.get_record_batch(i)["text"].to_pylist()[0]
                for i in indices
            ]
            return self.collator(raw_samples)

        loader = DataLoader(
            train_index_dataset,
            batch_size=self.batch_size,
            collate_fn=train_pl_collate_fn,
            num_workers=self.num_cpu_worker,
            pin_memory=True,
            drop_last=False,
        )
        self.logger.info("Finished loading training data")
        return loader

    def val_dataloader(self):

        valid_set_size = self._valid_dataset.num_record_batches
        valid_indexes = list(range(valid_set_size))

        if torch.distributed.is_initialized():
            global_rank = torch.distributed.get_rank()
        else:
            global_rank = 0

        world_size = torch.cuda.device_count()

        local_rank = global_rank % world_size  # TODO CONFIGRUABEL

        min_num_samples = torch.tensor(valid_set_size, device=f"cuda:{local_rank}")

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                min_num_samples, op=torch.distributed.ReduceOp.MIN
            )
        min_num_samples = min_num_samples.item()
        valid_indexes = valid_indexes[:min_num_samples]

        valid_index_dataset = IndexDataset(valid_indexes)

        print(
            f"### load valid set with size {min_num_samples} from {valid_set_size} samples on rank {self.global_rank}"
        )

        def val_pl_collate_fn(indices):
            inputs = [
                self._valid_dataset.get_record_batch(i)["text"].to_pylist()[0]
                for i in indices
            ]
            return self.collator(inputs)

        loader = DataLoader(
            valid_index_dataset,
            batch_size=self.batch_size,
            collate_fn=val_pl_collate_fn,
            num_workers=self.num_cpu_worker,
            pin_memory=True,
            drop_last=False,
        )
        self.logger.info(f"Finished loading validation data")

        return loader

    def val_nas_dataloader(self):

        valid_set_size = self._valid_nas_dataset.num_record_batches
        valid_indexes = list(range(valid_set_size))

        if torch.distributed.is_initialized():
            global_rank = torch.distributed.get_rank()
        else:
            global_rank = 0

        world_size = torch.cuda.device_count()

        local_rank = global_rank % world_size

        min_num_samples = torch.tensor(valid_set_size, device=f"cuda:{local_rank}")

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                min_num_samples, op=torch.distributed.ReduceOp.MIN
            )
        min_num_samples = min_num_samples.item()
        valid_indexes = valid_indexes[:min_num_samples]

        valid_index_dataset = IndexDataset(valid_indexes)

        print(
            f"### load valid set with size {min_num_samples} from {valid_set_size} samples on rank {self.global_rank}"
        )

        def val_pl_collate_fn(indices):
            inputs = [
                self._valid_nas_dataset.get_record_batch(i)["text"].to_pylist()[0]
                for i in indices
            ]
            return self.collator(inputs)

        loader = DataLoader(
            valid_index_dataset,
            batch_size=self.batch_size,
            collate_fn=val_pl_collate_fn,
            num_workers=self.num_cpu_worker,
            pin_memory=True,
            drop_last=False,
        )
        self.logger.info(f"Finished loading validation data")

        return loader


if __name__ == "__main__":
    """
    TEST DATA MODULE local
    """
    import logging, os
    import numpy as np

    print(os.environ)

    lm_data_config = {
        "dataset_name": "tinystories",
        "num_cpu_worker": 24,
        "num_gpu_worker": 4,
        "max_sample_len": 1024,
        "seed": 1,
        "batch_size": 8,
        "val_ratio": 0.25,
        "val_split_seed": 2357,
        "data_dir": "/path/to/datasets/",
        "cache_dir": "/path/to/datasets/cache",
    }

    my_data_handler = PlArrowFileModule(**lm_data_config)

    # os.environ["MASTER_ADDR"] = '127.0.0.1'
    # os.environ["MASTER_PORT"] = str(29500)
    # torch.distributed.init_process_group("nccl", rank=0, world_size=1)

    my_data_handler.prepare_data()
    my_data_handler.setup(0)

    print("data prepared")
    train_dataloader = my_data_handler.val_nas_dataloader()
    for i, item in enumerate(train_dataloader):
        print(item["src_len"].shape)
        print(item["trg_len"].shape)
        print(item["src_seq"].shape)
        print(item["trg_seq"].shape)
        print(item["src_seq"])
        print(item["trg_len"])
        print(item["trg_seq"])
        break
