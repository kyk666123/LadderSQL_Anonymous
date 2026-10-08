# Copyright (c) Microsoft. All rights reserved.

# type: ignore
import os
import json
import torch
from datasets import Dataset as HuggingFaceDataset
from omegaconf import DictConfig
from verl.utils.dataset.rl_dataset import RLHFDataset

from typing import Iterator, Optional
from torch.utils.data import IterableDataset

from agentlightning.types import Dataset

__all__ = [
    "AgentDataset",
    "LoadedDataset",
]


class AgentDataset(RLHFDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.filter_overlong_prompts = False

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        # Workaround for data proto. At least one tensor is needed.
        row_dict["fake_ids"] = torch.ones(1, dtype=torch.int)
        return row_dict


class LoadedDataset(AgentDataset):

    def __init__(self, dataset: Dataset):
        super().__init__([], None, DictConfig({}))  # type: ignore
        dataset_copy = [dataset[i] for i in range(len(dataset))]
        self.dataframe = HuggingFaceDataset.from_list(dataset_copy)

    def _read_files_and_tokenize(self):
        pass


#20260309修改
class DynamicVarianceDataset(AgentDataset, IterableDataset):
    """
    基于 AgentDataset 的动态加权 IterableDataset。

    特点：
    1. 有放回采样
    2. 外部维护 variance_dict
    3. dataset 内记录每次采样时的：
       - sample_key
       - sample_idx
       - 当前权重
       - 当前排名（最大权重 rank=1）
    4. 采样日志实时 append 到 jsonl 文件
    """

    def __init__(
        self,
        data_files,
        tokenizer,
        processor=None,
        config=None,
        max_samples: int = -1,
        variance_dict: Optional[dict[str, float]] = None,
        sample_log_path: Optional[str] = None,
    ):
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config,
            max_samples=max_samples,
        )

        self.samples_per_epoch = len(self.dataframe)
        self.replacement = True

        self.generator = torch.Generator()
        seed = config.get("seed", None) if config is not None else None
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)

        self.key_to_index: dict[str, int] = {}
        self.index_to_key: list[str] = []
        self._build_key_index()

        if variance_dict is None:
            raise ValueError("variance_dict must be provided externally.")

        self._validate_external_variance_dict(variance_dict)
        self.variance_dict = variance_dict

        # 记录采样日志
        self.sample_log_path = sample_log_path
        self.sample_step = 0
        self.batch_step = 0
        self._current_batch_pos = 0

        if self.sample_log_path is not None:
            os.makedirs(os.path.dirname(self.sample_log_path), exist_ok=True)

    def __len__(self):
        return self.samples_per_epoch

    def _extract_sample_key_from_row(self, row_dict: dict) -> str:
        if "question" not in row_dict:
            raise KeyError(
                f"Expected key 'question' in row_dict, but got keys={list(row_dict.keys())}"
            )
        if "db_id" not in row_dict:
            raise KeyError(
                f"Expected key 'db_id' in row_dict, but got keys={list(row_dict.keys())}"
            )

        question = row_dict["question"]
        db_id = row_dict["db_id"]

        if not isinstance(question, str):
            question = str(question)
        if not isinstance(db_id, str):
            db_id = str(db_id)

        return f"{question}|{db_id}"

    def _build_key_index(self):
        n = len(self.dataframe)

        for idx in range(n):
            row_dict = self.dataframe[idx]
            sample_key = self._extract_sample_key_from_row(row_dict)

            if sample_key in self.key_to_index:
                raise ValueError(
                    "Duplicate sample key detected in dataset.\n"
                    f"Duplicated key: {repr(sample_key[:200])}"
                )

            self.key_to_index[sample_key] = idx
            self.index_to_key.append(sample_key)

    def _validate_external_variance_dict(self, variance_dict: dict[str, float]):
        if not isinstance(variance_dict, dict):
            raise TypeError("variance_dict must be a dict[str, float]")

        missing_keys = []
        weights = []

        for sample_key in self.index_to_key:
            if sample_key not in variance_dict:
                missing_keys.append(sample_key)
                continue

            val = variance_dict[sample_key]
            try:
                val = float(val)
            except Exception as e:
                raise ValueError(
                    f"variance_dict[{repr(sample_key[:100])}] cannot be converted to float: {val}"
                ) from e

            if val < 0:
                raise ValueError(
                    f"variance_dict[{repr(sample_key[:100])}] is negative: {val}"
                )

            weights.append(val)

        if missing_keys:
            preview = [repr(k[:100]) for k in missing_keys[:3]]
            raise ValueError(
                "External variance_dict does not cover all dataset sample keys. "
                f"Missing count={len(missing_keys)}, examples={preview}"
            )

        if sum(weights) <= 0:
            raise ValueError("All initial weights in variance_dict are zero. Cannot sample.")

    def _build_weight_tensor_from_variance_dict(self) -> torch.Tensor:
        weights = []

        for sample_key in self.index_to_key:
            if sample_key not in self.variance_dict:
                raise KeyError(
                    "Sample key missing in current variance_dict during sampling: "
                    f"{repr(sample_key[:200])}"
                )

            val = self.variance_dict[sample_key]
            try:
                val = float(val)
            except Exception as e:
                raise ValueError(
                    f"variance_dict[{repr(sample_key[:100])}] cannot be converted to float: {val}"
                ) from e

            if val < 0:
                raise ValueError(
                    f"variance_dict[{repr(sample_key[:100])}] is negative: {val}"
                )

            weights.append(val)

        weight_tensor = torch.tensor(weights, dtype=torch.float64)

        if weight_tensor.sum().item() <= 0:
            raise ValueError("All weights in variance_dict are zero. Cannot sample.")

        return weight_tensor

    def _get_rank_of_index(self, weights: torch.Tensor, idx: int) -> int:
        """
        排名定义：最大权重 rank=1。
        并列时采用竞争排名，例如并列最大都为 1。
        """
        target_weight = weights[idx]
        rank = 1 + int((weights > target_weight).sum().item())
        return rank

    def _append_jsonl(self, path: str, record: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def start_new_batch(self, batch_step: Optional[int] = None):
        """
        由外部训练循环在每个 batch 开始前调用。
        这样 dataset 记录采样日志时，能知道当前是第几个 batch。
        """
        if batch_step is None:
            self.batch_step += 1
        else:
            self.batch_step = batch_step
        self._current_batch_pos = 0

    def __iter__(self) -> Iterator[dict]:
        produced = 0

        while produced < self.samples_per_epoch:
            weights = self._build_weight_tensor_from_variance_dict()

            sampled_idx = torch.multinomial(
                weights,
                num_samples=1,
                replacement=self.replacement,
                generator=self.generator,
            ).item()

            row = self.__getitem__(sampled_idx)

            sample_key = self.index_to_key[sampled_idx]
            current_weight = float(weights[sampled_idx].item())
            current_rank = self._get_rank_of_index(weights, sampled_idx)

            row["sample_idx"] = torch.tensor(sampled_idx, dtype=torch.long)
            row["sample_key"] = sample_key
            row["sample_weight"] = torch.tensor(current_weight, dtype=torch.float32)
            row["sample_rank"] = torch.tensor(current_rank, dtype=torch.long)

            if self.sample_log_path is not None:
                self._append_jsonl(
                    self.sample_log_path,
                    {
                        "type": "sample",
                        "sample_step": self.sample_step,
                        "batch_step": self.batch_step,
                        "sample_in_batch": self._current_batch_pos,
                        "sample_idx": sampled_idx,
                        "sample_key": sample_key,
                        "weight": current_weight,
                        "rank": current_rank,
                    },
                )

            self.sample_step += 1
            self._current_batch_pos += 1
            produced += 1
            yield row

#20260309修改