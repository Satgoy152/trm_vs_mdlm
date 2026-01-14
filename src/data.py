"""Data loading utilities for FineWeb-Edu streaming."""

from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer


class PackedDataset(IterableDataset):
    """
    Streams FineWeb-Edu, tokenizes, and packs into fixed-length sequences.

    Multiple documents are concatenated with EOS tokens and chunked
    into seq_len sequences for efficient training.
    """

    def __init__(
        self,
        dataset_name: str,
        subset: str,
        tokenizer_name: str,
        seq_len: int,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_name = dataset_name
        self.subset = subset
        self.seq_len = seq_len
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.eos_token_id = self.tokenizer.eos_token_id

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        """Yield packed sequences of token ids."""
        dataset = load_dataset(
            self.dataset_name,
            self.subset,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        dataset = dataset.shuffle(seed=self.seed, buffer_size=10_000)

        # Shard for distributed training
        if self.world_size > 1:
            dataset = dataset.shard(num_shards=self.world_size, index=self.rank)

        buffer: list[int] = []

        for example in dataset:
            text = example["text"]
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            tokens.append(self.eos_token_id)
            buffer.extend(tokens)

            # Yield complete sequences from buffer
            while len(buffer) >= self.seq_len:
                input_ids = torch.tensor(buffer[: self.seq_len], dtype=torch.long)
                attention_mask = torch.ones(self.seq_len, dtype=torch.long)
                buffer = buffer[self.seq_len :]

                yield {"input_ids": input_ids, "attention_mask": attention_mask}


def get_dataloader(config: dict, accelerator=None) -> DataLoader:
    """
    Create an infinite streaming dataloader for FineWeb-Edu.

    Args:
        config: Full configuration dictionary
        accelerator: HuggingFace Accelerator instance (optional)

    Returns:
        DataLoader yielding {"input_ids": [B, L], "attention_mask": [B, L]}
    """
    rank = 0
    world_size = 1
    if accelerator is not None:
        rank = accelerator.process_index
        world_size = accelerator.num_processes

    dataset = PackedDataset(
        dataset_name=config["data"]["dataset"],
        subset=config["data"]["subset"],
        tokenizer_name=config["data"]["tokenizer"],
        seq_len=config["data"]["seq_len"],
        seed=config["training"]["seed"],
        rank=rank,
        world_size=world_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=0,  # Streaming datasets work best with 0 workers
        pin_memory=True,
    )

    return dataloader


if __name__ == "__main__":
    # Test the dataloader
    import yaml

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Use smaller batch for testing
    config["training"]["batch_size"] = 2

    dataloader = get_dataloader(config)

    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  Sample tokens: {batch['input_ids'][0, :20].tolist()}")

        if i >= 2:
            break

    print("\nDataloader test passed!")
