"""Masked Diffusion Language Model (MDLM) training wrapper."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model import BidirectionalGPT2, Embeddings, OutputHead


class MDLM(nn.Module):
    """
    Masked Diffusion Language Model.

    Single-pass masked prediction with random mask ratios sampled
    uniformly between min and max.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        model_cfg = config["model"]
        mdlm_cfg = config["mdlm"]

        self.mask_ratio_min = mdlm_cfg["mask_ratio_min"]
        self.mask_ratio_max = mdlm_cfg["mask_ratio_max"]
        self.mask_token_id = config["data"]["mask_token_id"]

        self.embeddings = Embeddings(
            vocab_size=model_cfg["vocab_size"],
            d_model=model_cfg["d_model"],
            max_seq_len=model_cfg["max_seq_len"],
            dropout=model_cfg["dropout"],
        )
        self.backbone = BidirectionalGPT2(config)
        self.output_head = OutputHead(
            d_model=model_cfg["d_model"],
            vocab_size=model_cfg["vocab_size"],
            tie_weights=self.embeddings.token_embed.weight,
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> dict:
        """
        Single training step.

        1. Sample mask_ratio ~ U(min, max)
        2. Create mask at random positions
        3. Replace masked tokens with mask_token_id
        4. Forward through backbone
        5. Compute CE loss on masked positions only

        Args:
            input_ids: [B, L] token ids
            attention_mask: [B, L] padding mask

        Returns:
            {"loss": tensor, "accuracy": tensor, "mask_ratio": float}
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Sample mask ratio
        mask_ratio = torch.empty(1).uniform_(self.mask_ratio_min, self.mask_ratio_max).item()

        # Create mask (True = masked)
        rand = torch.rand(B, L, device=device)
        mask = (rand < mask_ratio) & (attention_mask == 1)

        # Store original tokens before masking
        targets = input_ids.clone()

        # Apply mask
        masked_input = input_ids.clone()
        masked_input[mask] = self.mask_token_id

        # Forward pass
        x = self.embeddings(masked_input)
        hidden = self.backbone(x, attention_mask)
        logits = self.output_head(hidden)

        # Compute loss only on masked positions
        masked_logits = logits[mask]  # [num_masked, V]
        masked_targets = targets[mask]  # [num_masked]

        if masked_logits.numel() == 0:
            # Edge case: no tokens masked
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            accuracy = torch.tensor(1.0, device=device)
        else:
            loss = F.cross_entropy(masked_logits, masked_targets)

            # Compute accuracy
            preds = masked_logits.argmax(dim=-1)
            accuracy = (preds == masked_targets).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
            "mask_ratio": mask_ratio,
        }

    @torch.no_grad()
    def generate(self, batch_size: int, seq_len: int, steps: int = 10) -> Tensor:
        """
        Iterative unmasking generation.

        Starts with all tokens masked, then iteratively unmasks
        the highest-confidence predictions.

        Args:
            batch_size: Number of sequences to generate
            seq_len: Length of sequences
            steps: Number of unmasking steps

        Returns:
            generated: [B, L] generated token ids
        """
        device = next(self.parameters()).device

        # Start with all masked
        tokens = torch.full(
            (batch_size, seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        # Iterative unmasking
        for step in range(steps):
            # Forward pass
            x = self.embeddings(tokens)
            hidden = self.backbone(x, attention_mask)
            logits = self.output_head(hidden)

            # Get confidence scores and predictions
            probs = F.softmax(logits, dim=-1)
            confidence, preds = probs.max(dim=-1)  # [B, L]

            # Find masked positions
            is_masked = tokens == self.mask_token_id

            # Set confidence to -inf for non-masked positions
            confidence = confidence.masked_fill(~is_masked, -float("inf"))

            # Compute how many tokens to unmask this step
            num_masked = is_masked.sum(dim=-1)  # [B]
            tokens_per_step = (seq_len + steps - 1) // steps
            num_to_unmask = torch.minimum(
                torch.full_like(num_masked, tokens_per_step),
                num_masked,
            )

            # Unmask top-k most confident tokens per sequence
            for b in range(batch_size):
                if num_to_unmask[b] > 0:
                    k = num_to_unmask[b].item()
                    _, top_indices = confidence[b].topk(k)
                    tokens[b, top_indices] = preds[b, top_indices]

        return tokens


if __name__ == "__main__":
    # Test MDLM
    import yaml

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    B, L = 2, 512
    V = config["model"]["vocab_size"]

    model = MDLM(config)

    # Test forward
    input_ids = torch.randint(0, V, (B, L))
    attention_mask = torch.ones(B, L)

    outputs = model(input_ids, attention_mask)
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Accuracy: {outputs['accuracy'].item():.4f}")
    print(f"Mask ratio: {outputs['mask_ratio']:.4f}")

    assert outputs["loss"].isfinite()
    assert 0 <= outputs["accuracy"] <= 1

    # Test backward
    outputs["loss"].backward()
    print("Backward pass successful")

    # Test generation (small for speed)
    generated = model.generate(batch_size=1, seq_len=32, steps=5)
    print(f"Generated shape: {generated.shape}")
    assert generated.shape == (1, 32)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\nMDLM test passed!")
