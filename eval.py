"""
Evaluation script for MDLM model.

Usage:
    python eval.py --checkpoint path/to/checkpoint
"""

import argparse

import torch
import yaml
from accelerate import Accelerator
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data import get_dataloader
from src.mdlm import MDLM


def evaluate_reconstruction(
    model: torch.nn.Module,
    dataloader: DataLoader,
    mask_token_id: int,
    mask_ratios: list[float] = [0.15, 0.3, 0.5, 0.7],
    num_batches: int = 100,
    device: str = "cuda",
) -> dict:
    """
    Evaluate reconstruction quality at various mask ratios.

    Args:
        model: TRM or MDLM model
        dataloader: Evaluation dataloader
        mask_token_id: Token ID used for masking
        mask_ratios: List of mask ratios to evaluate
        num_batches: Number of batches to evaluate
        device: Device to run on

    Returns:
        Dictionary of metrics per mask ratio
    """
    model.eval()
    results = {}

    for ratio in mask_ratios:
        total_loss = 0.0
        total_correct = 0
        total_masked = 0

        batch_iter = iter(dataloader)
        for _ in tqdm(range(num_batches), desc=f"Evaluating mask_ratio={ratio}"):
            try:
                batch = next(batch_iter)
            except StopIteration:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            B, L = input_ids.shape

            # Create mask
            rand = torch.rand(B, L, device=device)
            mask = (rand < ratio) & (attention_mask == 1)

            # Store targets
            targets = input_ids.clone()

            # Apply mask
            masked_input = input_ids.clone()
            masked_input[mask] = mask_token_id

            with torch.no_grad():
                # Get embeddings and forward
                x = model.embeddings(masked_input)
                hidden = model.backbone(x, attention_mask)
                logits = model.output_head(hidden)

                # Compute metrics on masked positions
                masked_logits = logits[mask]
                masked_targets = targets[mask]

                if masked_logits.numel() > 0:
                    loss = torch.nn.functional.cross_entropy(
                        masked_logits, masked_targets, reduction="sum"
                    )
                    preds = masked_logits.argmax(dim=-1)
                    correct = (preds == masked_targets).sum().item()

                    total_loss += loss.item()
                    total_correct += correct
                    total_masked += masked_logits.size(0)

        if total_masked > 0:
            results[ratio] = {
                "loss": total_loss / total_masked,
                "accuracy": total_correct / total_masked,
                "perplexity": torch.exp(torch.tensor(total_loss / total_masked)).item(),
                "num_tokens": total_masked,
            }
        else:
            results[ratio] = {"loss": 0, "accuracy": 0, "perplexity": 0, "num_tokens": 0}

    return results


def generate_samples(
    model: torch.nn.Module,
    tokenizer,
    n_samples: int = 5,
    seq_len: int = 128,
    steps: int = 10,
    device: str = "cuda",
) -> list[str]:
    """
    Generate text samples from scratch.

    Args:
        model: MDLM model
        tokenizer: Tokenizer for decoding
        n_samples: Number of samples to generate
        seq_len: Length of generated sequences
        steps: Number of unmasking/refinement steps
        device: Device to run on

    Returns:
        List of decoded text samples
    """
    model.eval()
    model.to(device)

    generated = model.generate(batch_size=n_samples, seq_len=seq_len, steps=steps)
    samples = []

    for i in range(n_samples):
        tokens = generated[i].tolist()
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        samples.append(text)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate MDLM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=100,
        help="Number of batches for evaluation",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--gen_steps",
        type=int,
        default=10,
        help="Number of generation steps",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    model = MDLM(config)

    # Load checkpoint
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    accelerator.load_state(args.checkpoint)
    model = accelerator.unwrap_model(model)
    model.to(device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Get dataloader for evaluation
    config["training"]["batch_size"] = 8  # Smaller batch for eval
    dataloader = get_dataloader(config)

    # Run reconstruction evaluation
    print("\n" + "=" * 50)
    print("Reconstruction Evaluation")
    print("=" * 50)

    mask_ratios = [0.15, 0.3, 0.5, 0.7]
    results = evaluate_reconstruction(
        model=model,
        dataloader=dataloader,
        mask_token_id=config["data"]["mask_token_id"],
        mask_ratios=mask_ratios,
        num_batches=args.num_batches,
        device=device,
    )

    print(f"\n{'Mask Ratio':<12} {'Loss':<10} {'Accuracy':<10} {'Perplexity':<12}")
    print("-" * 44)
    for ratio, metrics in results.items():
        print(
            f"{ratio:<12.2f} {metrics['loss']:<10.4f} {metrics['accuracy']:<10.4f} {metrics['perplexity']:<12.2f}"
        )

    # Generate samples
    print("\n" + "=" * 50)
    print("Generated Samples")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer"])
    samples = generate_samples(
        model=model,
        tokenizer=tokenizer,
        n_samples=args.n_samples,
        seq_len=128,
        steps=args.gen_steps,
        device=device,
    )

    for i, sample in enumerate(samples):
        print(f"\nSample {i + 1}:")
        print("-" * 40)
        print(sample[:500])  # Truncate for readability
        if len(sample) > 500:
            print("...")

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Method: {args.method}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Best accuracy (mask_ratio=0.15): {results[0.15]['accuracy']:.4f}")
    print(f"Perplexity at 50% masking: {results[0.5]['perplexity']:.2f}")


if __name__ == "__main__":
    main()
