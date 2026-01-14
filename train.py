"""
Unified training script for TRM vs MDLM comparison.

Usage:
    accelerate launch train.py --method trm
    accelerate launch train.py --method mdlm
"""

import argparse
import math
import os
import time

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.data import get_dataloader
from src.mdlm import MDLM
from src.trm import TRM


def get_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Create cosine scheduler with linear warmup."""
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-6,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )


def main():
    parser = argparse.ArgumentParser(description="Train TRM or MDLM")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["trm", "mdlm"],
        help="Training method to use",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg["grad_accum_steps"],
        log_with="wandb",
        mixed_precision="bf16",
    )

    # Set seed for reproducibility
    set_seed(train_cfg["seed"])

    # Create model
    if args.method == "trm":
        model = TRM(config)
    else:
        model = MDLM(config)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Method: {args.method}")
    accelerator.print(f"Total parameters: {total_params:,}")

    # Create dataloader
    dataloader = get_dataloader(config, accelerator)

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        betas=(0.9, 0.95),
    )

    # Calculate total steps
    tokens_per_step = (
        train_cfg["batch_size"]
        * config["data"]["seq_len"]
        * train_cfg["grad_accum_steps"]
        * accelerator.num_processes
    )
    total_steps = train_cfg["total_tokens"] // tokens_per_step + 1

    # Create scheduler
    scheduler = get_scheduler(optimizer, train_cfg["warmup_steps"], total_steps)

    # Prepare with accelerator
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Initialize wandb
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="trm-vs-mdlm",
            config=config,
            init_kwargs={"wandb": {"name": f"{args.method}-{time.strftime('%Y%m%d-%H%M%S')}"}},
        )

    # Resume from checkpoint if specified
    start_step = 0
    tokens_seen = 0
    if args.resume:
        accelerator.load_state(args.resume)
        # Try to restore step count from checkpoint path
        if "step_" in args.resume:
            start_step = int(args.resume.split("step_")[-1])
            tokens_seen = start_step * tokens_per_step
        accelerator.print(f"Resumed from step {start_step}")

    # Training loop
    start_time = time.time()
    step = start_step
    model.train()

    accelerator.print(f"Starting training from step {step}")
    accelerator.print(f"Total tokens target: {train_cfg['total_tokens']:,}")
    accelerator.print(f"Tokens per step: {tokens_per_step:,}")
    accelerator.print(f"Total steps: {total_steps:,}")

    for batch in dataloader:
        with accelerator.accumulate(model):
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss = outputs["loss"]
            accelerator.backward(loss)

            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Update tokens seen
        if accelerator.sync_gradients:
            step += 1
            tokens_seen += tokens_per_step

            # Logging
            if step % train_cfg["log_every"] == 0:
                wall_time = time.time() - start_time
                tokens_per_sec = tokens_seen / wall_time if wall_time > 0 else 0

                log_dict = {
                    "loss": loss.item(),
                    "accuracy": outputs["accuracy"].item(),
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_seen": tokens_seen,
                    "wall_time": wall_time,
                    "tokens_per_sec": tokens_per_sec,
                    "step": step,
                }

                # Add method-specific metrics
                if args.method == "trm" and "n_sup_steps" in outputs:
                    log_dict["n_sup_steps"] = outputs["n_sup_steps"]
                if args.method == "mdlm" and "mask_ratio" in outputs:
                    log_dict["mask_ratio"] = outputs["mask_ratio"]

                accelerator.log(log_dict, step=step)

                accelerator.print(
                    f"Step {step} | Loss: {loss.item():.4f} | "
                    f"Acc: {outputs['accuracy'].item():.4f} | "
                    f"Tokens: {tokens_seen:,} | "
                    f"Time: {wall_time:.1f}s | "
                    f"Tok/s: {tokens_per_sec:.0f}"
                )

            # Checkpointing
            if step % train_cfg["save_every"] == 0:
                checkpoint_dir = f"checkpoints/{args.method}/step_{step}"
                accelerator.save_state(checkpoint_dir)
                accelerator.print(f"Saved checkpoint to {checkpoint_dir}")

            # Termination
            if tokens_seen >= train_cfg["total_tokens"]:
                accelerator.print(f"Reached {tokens_seen:,} tokens, stopping training")
                break

    # Final save
    final_checkpoint = f"checkpoints/{args.method}/final"
    accelerator.save_state(final_checkpoint)
    accelerator.print(f"Saved final checkpoint to {final_checkpoint}")

    # End tracking
    accelerator.end_training()

    total_time = time.time() - start_time
    accelerator.print(f"Training complete!")
    accelerator.print(f"Total time: {total_time:.1f}s")
    accelerator.print(f"Total tokens: {tokens_seen:,}")
    accelerator.print(f"Average tokens/sec: {tokens_seen / total_time:.0f}")


if __name__ == "__main__":
    main()
