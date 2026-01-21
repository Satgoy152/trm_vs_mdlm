"""
Training script for MDLM (Masked Diffusion Language Model).

Usage:
    accelerate launch train.py
"""

import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.data import get_dataloader
from src.mdlm import MDLM


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


def estimate_flops_per_step(config: dict) -> int:
    """
    Estimate FLOPs per training step for MDLM.
    """
    model_cfg = config["model"]
    n_layers = model_cfg["n_layers"]
    d_model = model_cfg["d_model"]
    d_ff = model_cfg["d_ff"]
    seq_len = config["data"]["seq_len"]
    batch_size = config["training"]["batch_size"]
    grad_accum = config["training"]["grad_accum_steps"]

    # Per-token FLOPs for one forward pass through transformer
    # Attention: 4 * L * D (Q, K, V projections + output)
    # FFN: 2 * D * d_ff
    # Total per layer: ~12 * D^2 + 8 * D * d_ff (approximate)
    flops_per_token_per_layer = 12 * d_model * d_model + 8 * d_model * d_ff

    # Attention FLOPs: 2 * L * D per head for QK^T and attention @ V
    attn_flops_per_token = 4 * seq_len * d_model

    flops_per_token = n_layers * (flops_per_token_per_layer + attn_flops_per_token)

    # Forward + backward = 3x forward FLOPs (approximately)
    flops_per_token *= 3

    # Total tokens per step
    tokens_per_step = batch_size * seq_len * grad_accum

    base_flops = flops_per_token * tokens_per_step

    return base_flops


@torch.no_grad()
def evaluate(model, eval_dataloader, config: dict, accelerator, num_batches: int) -> dict:
    """Run evaluation for MDLM (masked language modeling)."""
    model.eval()
    mask_token_id = config["data"]["mask_token_id"]

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    eval_iter = iter(eval_dataloader)
    for _ in range(num_batches):
        try:
            batch = next(eval_iter)
        except StopIteration:
            break

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        B, L = input_ids.shape

        # Use fixed mask ratio for eval consistency
        mask_ratio = 0.15
        rand = torch.rand(B, L, device=input_ids.device)
        mask = (rand < mask_ratio) & (attention_mask == 1)

        targets = input_ids.clone()
        masked_input = input_ids.clone()
        masked_input[mask] = mask_token_id

        # Forward pass (eval uses standard single pass)
        m = model.module if hasattr(model, 'module') else model
        x = m.embeddings(masked_input)
        hidden = m.backbone(x, attention_mask)
        logits = m.output_head(hidden)

        masked_logits = logits[mask]
        masked_targets = targets[mask]

        if masked_logits.numel() > 0:
            loss = F.cross_entropy(masked_logits, masked_targets, reduction="sum")
            preds = masked_logits.argmax(dim=-1)
            correct = (preds == masked_targets).sum()

            total_loss += accelerator.gather(loss).sum().item()
            total_correct += accelerator.gather(correct).sum().item()
            total_tokens += accelerator.gather(torch.tensor(masked_logits.size(0), device=input_ids.device)).sum().item()

    model.train()

    if total_tokens > 0:
        return {
            "eval_loss": total_loss / total_tokens,
            "eval_accuracy": total_correct / total_tokens,
            "eval_perplexity": torch.exp(torch.tensor(total_loss / total_tokens)).item(),
        }
    return {"eval_loss": 0, "eval_accuracy": 0, "eval_perplexity": 0}


class CSVLogger:
    """Simple CSV logger for training metrics."""

    def __init__(self, filepath: str, fieldnames: list[str]):
        self.filepath = filepath
        self.fieldnames = fieldnames

        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Write header
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row: dict):
        """Append a row to the CSV."""
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Train MDLM")
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for logs and checkpoints",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]

    # Handle DDP unused parameters for 2-step MDLM training
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg["grad_accum_steps"],
        log_with="wandb",
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs],
    )

    # Set seed for reproducibility
    set_seed(train_cfg["seed"])

    # Create model
    model = MDLM(config)

    # Calculate FLOPs per step
    flops_per_step = estimate_flops_per_step(config)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Total parameters: {total_params:,}")
    accelerator.print(f"Estimated FLOPs per step: {flops_per_step:,}")

    # Create dataloaders
    dataloader = get_dataloader(config, accelerator)
    eval_dataloader = get_dataloader(config, accelerator)  # Separate iterator for eval

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
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
    eval_dataloader = accelerator.prepare(eval_dataloader)

    # Initialize wandb
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="mdlm-training",
            config=config,
            init_kwargs={"wandb": {"name": f"mdlm-{time.strftime('%Y%m%d-%H%M%S')}"}},
        )

    # Setup CSV logger (main process only)
    csv_logger = None
    if accelerator.is_main_process:
        csv_path = f"{args.output_dir}/mdlm/metrics.csv"
        csv_logger = CSVLogger(
            csv_path,
            fieldnames=[
                "step", "tokens_seen", "flops", "wall_time",
                "train_loss", "train_accuracy", 
                "train_ent_masked_1", "train_ent_unmasked_1",
                "train_ent_masked_2", "train_ent_unmasked_2",
                "lr",
                "eval_loss", "eval_accuracy", "eval_perplexity",
            ],
        )
        accelerator.print(f"Logging metrics to {csv_path}")

    # Resume from checkpoint if specified
    start_step = 0
    tokens_seen = 0
    total_flops = 0
    if args.resume:
        accelerator.load_state(args.resume)
        if "step_" in args.resume:
            start_step = int(args.resume.split("step_")[-1])
            tokens_seen = start_step * tokens_per_step
            total_flops = start_step * flops_per_step
        accelerator.print(f"Resumed from step {start_step}")

    # Training loop
    start_time = time.time()
    step = start_step
    model.train()

    accelerator.print(f"Starting training from step {step}")
    accelerator.print(f"Total tokens target: {train_cfg['total_tokens']:,}")
    accelerator.print(f"Tokens per step: {tokens_per_step:,}")
    accelerator.print(f"Total steps: {total_steps:,}")
    accelerator.print(f"Log train every: {train_cfg['log_train_every']} steps")
    accelerator.print(f"Log eval every: {train_cfg['log_eval_every']} steps")

    # Track recent losses for smoothing
    recent_losses = []
    recent_accuracies = []
    recent_ent_masked_1 = []
    recent_ent_unmasked_1 = []
    recent_ent_masked_2 = []
    recent_ent_unmasked_2 = []

    for batch in dataloader:
        with accelerator.accumulate(model):
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss = outputs["loss"]
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            step += 1
            tokens_seen += tokens_per_step
            total_flops += flops_per_step

            # Track recent metrics
            recent_losses.append(loss.item())
            recent_accuracies.append(outputs["accuracy"].item())
            
            # Additional entropy metrics
            recent_ent_masked_1.append(outputs["ent_masked_1"].item())
            recent_ent_unmasked_1.append(outputs["ent_unmasked_1"].item())
            recent_ent_masked_2.append(outputs["ent_masked_2"].item())
            recent_ent_unmasked_2.append(outputs["ent_unmasked_2"].item())
            
            if len(recent_losses) > 100:
                recent_losses.pop(0)
                recent_accuracies.pop(0)
                recent_ent_masked_1.pop(0)
                recent_ent_unmasked_1.pop(0)
                recent_ent_masked_2.pop(0)
                recent_ent_unmasked_2.pop(0)

            wall_time = time.time() - start_time

            # Log train metrics every log_train_every steps
            if step % train_cfg["log_train_every"] == 0:
                avg_loss = sum(recent_losses) / len(recent_losses)
                avg_acc = sum(recent_accuracies) / len(recent_accuracies)
                
                avg_ent_m1 = sum(recent_ent_masked_1) / len(recent_ent_masked_1)
                avg_ent_u1 = sum(recent_ent_unmasked_1) / len(recent_ent_unmasked_1)
                avg_ent_m2 = sum(recent_ent_masked_2) / len(recent_ent_masked_2)
                avg_ent_u2 = sum(recent_ent_unmasked_2) / len(recent_ent_unmasked_2)

                accelerator.print(
                    f"Step {step} | Loss: {avg_loss:.4f} | "
                    f"Acc: {avg_acc:.4f} | "
                    f"Ent(M1/U1): {avg_ent_m1:.2f}/{avg_ent_u1:.2f} | "
                    f"Tokens: {tokens_seen:,} | "
                    f"FLOPs: {total_flops:.2e}"
                )

                # Log to wandb
                log_dict = {
                    "train_loss": avg_loss,
                    "train_accuracy": avg_acc,
                    "train_ent_masked_1": avg_ent_m1,
                    "train_ent_unmasked_1": avg_ent_u1,
                    "train_ent_masked_2": avg_ent_m2,
                    "train_ent_unmasked_2": avg_ent_u2,
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_seen": tokens_seen,
                    "flops": total_flops,
                    "wall_time": wall_time,
                }
                accelerator.log(log_dict, step=step)

            # Log eval metrics every log_eval_every steps
            if step % train_cfg["log_eval_every"] == 0:
                eval_metrics = evaluate(
                    model, eval_dataloader, config, accelerator,
                    num_batches=train_cfg["eval_batches"]
                )

                accelerator.print(
                    f"  Eval | Loss: {eval_metrics['eval_loss']:.4f} | "
                    f"Acc: {eval_metrics['eval_accuracy']:.4f} | "
                    f"PPL: {eval_metrics['eval_perplexity']:.2f}"
                )

                # Log to wandb
                accelerator.log(eval_metrics, step=step)

                # Log to CSV (main process only)
                if csv_logger is not None:
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    avg_acc = sum(recent_accuracies) / len(recent_accuracies)
                    
                    avg_ent_m1 = sum(recent_ent_masked_1) / len(recent_ent_masked_1)
                    avg_ent_u1 = sum(recent_ent_unmasked_1) / len(recent_ent_unmasked_1)
                    avg_ent_m2 = sum(recent_ent_masked_2) / len(recent_ent_masked_2)
                    avg_ent_u2 = sum(recent_ent_unmasked_2) / len(recent_ent_unmasked_2)

                    csv_logger.log({
                        "step": step,
                        "tokens_seen": tokens_seen,
                        "flops": total_flops,
                        "wall_time": wall_time,
                        "train_loss": avg_loss,
                        "train_accuracy": avg_acc,
                        "train_ent_masked_1": avg_ent_m1,
                        "train_ent_unmasked_1": avg_ent_u1,
                        "train_ent_masked_2": avg_ent_m2,
                        "train_ent_unmasked_2": avg_ent_u2,
                        "lr": scheduler.get_last_lr()[0],
                        "eval_loss": eval_metrics["eval_loss"],
                        "eval_accuracy": eval_metrics["eval_accuracy"],
                        "eval_perplexity": eval_metrics["eval_perplexity"],
                    })

            # Checkpointing
            if step % train_cfg["save_every"] == 0:
                checkpoint_dir = f"{args.output_dir}/mdlm/checkpoints/step_{step}"
                accelerator.save_state(checkpoint_dir)
                accelerator.print(f"Saved checkpoint to {checkpoint_dir}")

            # Termination
            if tokens_seen >= train_cfg["total_tokens"]:
                accelerator.print(f"Reached {tokens_seen:,} tokens, stopping training")
                break

    # Final evaluation
    final_eval = evaluate(
        model, eval_dataloader, config, accelerator,
        num_batches=train_cfg["eval_batches"] * 2
    )
    accelerator.print(f"Final eval | Loss: {final_eval['eval_loss']:.4f} | "
                     f"Acc: {final_eval['eval_accuracy']:.4f}")

    # Final save
    final_checkpoint = f"{args.output_dir}/mdlm/checkpoints/final"
    accelerator.save_state(final_checkpoint)
    accelerator.print(f"Saved final checkpoint to {final_checkpoint}")

    # Log final metrics to CSV
    if csv_logger is not None:
        csv_logger.log({
            "step": step,
            "tokens_seen": tokens_seen,
            "flops": total_flops,
            "wall_time": time.time() - start_time,
            "train_loss": sum(recent_losses) / len(recent_losses) if recent_losses else 0,
            "train_accuracy": sum(recent_accuracies) / len(recent_accuracies) if recent_accuracies else 0,
            "train_ent_masked_1": sum(recent_ent_masked_1) / len(recent_ent_masked_1) if recent_ent_masked_1 else 0,
            "train_ent_unmasked_1": sum(recent_ent_unmasked_1) / len(recent_ent_unmasked_1) if recent_ent_unmasked_1 else 0,
            "train_ent_masked_2": sum(recent_ent_masked_2) / len(recent_ent_masked_2) if recent_ent_masked_2 else 0,
            "train_ent_unmasked_2": sum(recent_ent_unmasked_2) / len(recent_ent_unmasked_2) if recent_ent_unmasked_2 else 0,
            "lr": scheduler.get_last_lr()[0],
            "eval_loss": final_eval["eval_loss"],
            "eval_accuracy": final_eval["eval_accuracy"],
            "eval_perplexity": final_eval["eval_perplexity"],
        })

    accelerator.end_training()

    total_time = time.time() - start_time
    accelerator.print(f"\nTraining complete!")
    accelerator.print(f"Total time: {total_time:.1f}s")
    accelerator.print(f"Total tokens: {tokens_seen:,}")
    accelerator.print(f"Total FLOPs: {total_flops:.2e}")
    accelerator.print(f"Average tokens/sec: {tokens_seen / total_time:.0f}")


if __name__ == "__main__":
    main()
