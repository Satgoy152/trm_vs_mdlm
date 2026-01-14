"""Tiny Recursive Model (TRM) for masked language modeling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model import BidirectionalGPT2, Embeddings, OutputHead, QHead


class TRM(nn.Module):
    """
    Tiny Recursive Model adapted for language modeling.

    Iteratively refines predictions through recursive latent updates
    with deep supervision.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        model_cfg = config["model"]
        trm_cfg = config["trm"]

        self.n = trm_cfg["n"]  # latent recursions per deep step
        self.T = trm_cfg["T"]  # deep recursions
        self.N_sup = trm_cfg["N_sup"]  # supervision steps
        self.mask_ratio = trm_cfg["mask_ratio"]
        self.mask_token_id = config["data"]["mask_token_id"]
        self.d_model = model_cfg["d_model"]

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
        self.q_head = QHead(model_cfg["d_model"])

        # Learned initial states
        self.z_init = nn.Parameter(torch.randn(1, 1, model_cfg["d_model"]) * 0.02)
        self.y_init = nn.Parameter(torch.randn(1, 1, model_cfg["d_model"]) * 0.02)

    def latent_recursion(
        self,
        x: Tensor,
        y: Tensor,
        z: Tensor,
        n: int,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Refine latent z through n iterations, then update y.

        Args:
            x: [B, L, D] embedded input (frozen)
            y: [B, L, D] current prediction embeddings
            z: [B, L, D] current latent state
            n: number of latent iterations
            attention_mask: [B, L] optional padding mask

        Returns:
            (y, z): updated prediction and latent embeddings
        """
        # Latent refinement iterations
        for _ in range(n):
            z = self.backbone(x + y + z, attention_mask)

        # Output refinement (no x)
        y = self.backbone(y + z, attention_mask)

        return y, z

    def deep_recursion(
        self,
        x: Tensor,
        y: Tensor,
        z: Tensor,
        n: int,
        T: int,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        T-1 no_grad recursions, then 1 with grad.

        Args:
            x: [B, L, D] embedded input
            y: [B, L, D] prediction embeddings
            z: [B, L, D] latent state
            n: latent recursions per deep step
            T: number of deep recursions
            attention_mask: [B, L] padding mask

        Returns:
            (y, z, logits, halt_logit): detached y/z, final logits, halt logit
        """
        # T-1 iterations without grad
        if T > 1:
            with torch.no_grad():
                for _ in range(T - 1):
                    y, z = self.latent_recursion(x, y, z, n, attention_mask)

        # Final iteration with grad
        y, z = self.latent_recursion(x, y, z, n, attention_mask)

        # Compute outputs
        logits = self.output_head(y)
        halt_logit = self.q_head(y, attention_mask)

        # Detach for next supervision step
        return y.detach(), z.detach(), logits, halt_logit

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> dict:
        """
        Full TRM training step with deep supervision.

        Args:
            input_ids: [B, L] token ids
            attention_mask: [B, L] padding mask

        Returns:
            {"loss": tensor, "accuracy": tensor, "n_sup_steps": int}
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Create mask
        rand = torch.rand(B, L, device=device)
        mask = (rand < self.mask_ratio) & (attention_mask == 1)

        # Store targets
        targets = input_ids.clone()

        # Apply mask
        masked_input = input_ids.clone()
        masked_input[mask] = self.mask_token_id

        # Embed input
        x = self.embeddings(masked_input)

        # Initialize y and z (broadcast to batch and sequence)
        y = self.y_init.expand(B, L, -1).clone()
        z = self.z_init.expand(B, L, -1).clone()

        # Deep supervision
        total_loss = 0.0
        total_accuracy = 0.0
        actual_steps = 0

        for sup_step in range(self.N_sup):
            y, z, logits, halt_logit = self.deep_recursion(
                x, y, z, self.n, self.T, attention_mask
            )

            # Compute loss on masked positions
            masked_logits = logits[mask]
            masked_targets = targets[mask]

            if masked_logits.numel() > 0:
                ce_loss = F.cross_entropy(masked_logits, masked_targets)

                # Q-head loss (optional): predict if predictions are correct
                preds = masked_logits.argmax(dim=-1)
                target_accuracy = (preds == masked_targets).float().mean()
                q_loss = F.binary_cross_entropy_with_logits(
                    halt_logit.mean(), target_accuracy.detach()
                )

                step_loss = ce_loss + 0.1 * q_loss
                total_loss = total_loss + step_loss
                total_accuracy = total_accuracy + target_accuracy.detach()
                actual_steps += 1

                # Optional early stopping in training (when confident)
                halt_prob = torch.sigmoid(halt_logit.mean())
                if halt_prob > 0.5 and sup_step > 0:
                    break

        # Average over supervision steps
        if actual_steps > 0:
            loss = total_loss / actual_steps
            accuracy = total_accuracy / actual_steps
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            accuracy = torch.tensor(1.0, device=device)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "n_sup_steps": actual_steps,
        }

    @torch.no_grad()
    def generate(self, batch_size: int, seq_len: int, steps: int = 10) -> Tensor:
        """
        Generate sequences through iterative refinement.

        Args:
            batch_size: Number of sequences to generate
            seq_len: Length of sequences
            steps: Number of refinement steps

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

        # Initialize x, y, z
        x = self.embeddings(tokens)
        y = self.y_init.expand(batch_size, seq_len, -1).clone()
        z = self.z_init.expand(batch_size, seq_len, -1).clone()

        # Refine through steps
        for _ in range(steps):
            y, z = self.latent_recursion(x, y, z, self.n, attention_mask)

        # Get final predictions
        logits = self.output_head(y)
        tokens = logits.argmax(dim=-1)

        return tokens


if __name__ == "__main__":
    # Test TRM with small parameters first
    import yaml

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Use smaller values for testing
    config["trm"]["n"] = 1
    config["trm"]["T"] = 1
    config["trm"]["N_sup"] = 1

    B, L = 2, 64  # Smaller for testing
    V = config["model"]["vocab_size"]

    model = TRM(config)

    # Test forward
    input_ids = torch.randint(0, V, (B, L))
    attention_mask = torch.ones(B, L)

    outputs = model(input_ids, attention_mask)
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Accuracy: {outputs['accuracy'].item():.4f}")
    print(f"N_sup steps: {outputs['n_sup_steps']}")

    assert outputs["loss"].isfinite()
    assert 0 <= outputs["accuracy"] <= 1

    # Test backward
    outputs["loss"].backward()
    print("Backward pass successful")

    # Test with full parameters
    print("\nTesting with full parameters...")
    config["trm"]["n"] = 2
    config["trm"]["T"] = 2
    config["trm"]["N_sup"] = 2

    model_full = TRM(config)
    outputs_full = model_full(input_ids, attention_mask)
    print(f"Full Loss: {outputs_full['loss'].item():.4f}")
    print(f"Full N_sup steps: {outputs_full['n_sup_steps']}")

    outputs_full["loss"].backward()
    print("Full backward pass successful")

    # Test generation
    generated = model.generate(batch_size=1, seq_len=32, steps=3)
    print(f"Generated shape: {generated.shape}")
    assert generated.shape == (1, 32)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\nTRM test passed!")
