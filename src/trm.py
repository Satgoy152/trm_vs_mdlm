"""Tiny Recursive Model (TRM) for prefix-based sequence prediction.

Instead of masked language modeling, TRM learns to predict continuations:
- x: Input prefix (variable length)
- y: Prediction sequence (fixed length, iteratively refined)
- z: Latent reasoning state (fixed length, iteratively refined)

The model concatenates [x, y, z] and uses bidirectional attention to
refine y and z through multiple iterations before predicting tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model import BidirectionalGPT2, Embeddings, OutputHead, QHead


class TRM(nn.Module):
    """
    Tiny Recursive Model for sequence prediction.

    Given a prefix x, iteratively refines prediction embeddings y
    using latent state z to predict the continuation.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        model_cfg = config["model"]
        trm_cfg = config["trm"]

        self.n = trm_cfg["n"]  # latent recursions per deep step
        self.T = trm_cfg["T"]  # deep recursions
        self.N_sup = trm_cfg["N_sup"]  # supervision steps
        self.y_len = trm_cfg["y_len"]  # prediction sequence length
        self.z_len = trm_cfg["z_len"]  # latent reasoning state length
        self.min_prefix = trm_cfg["min_prefix"]  # minimum prefix length
        self.d_model = model_cfg["d_model"]
        self.max_seq_len = model_cfg["max_seq_len"]

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

        # Learned initial states with configurable lengths
        self.z_init = nn.Parameter(torch.randn(1, self.z_len, model_cfg["d_model"]) * 0.02)
        self.y_init = nn.Parameter(torch.randn(1, self.y_len, model_cfg["d_model"]) * 0.02)

        # Learned position embeddings for y and z (relative to their segments)
        self.y_pos = nn.Parameter(torch.randn(1, self.y_len, model_cfg["d_model"]) * 0.02)
        self.z_pos = nn.Parameter(torch.randn(1, self.z_len, model_cfg["d_model"]) * 0.02)

    def _create_attention_mask(self, prefix_len: int, batch_size: int, device: torch.device) -> Tensor:
        """Create attention mask for [x, y, z] concatenation."""
        total_len = prefix_len + self.y_len + self.z_len
        # All positions can attend to all positions (bidirectional)
        return torch.ones(batch_size, total_len, device=device)

    def latent_recursion(
        self,
        x: Tensor,
        y: Tensor,
        z: Tensor,
        n: int,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Refine latent z through n iterations, then update y.

        Args:
            x: [B, L_x, D] embedded prefix (frozen during recursion)
            y: [B, L_y, D] current prediction embeddings
            z: [B, L_z, D] current latent state
            n: number of latent iterations
            attention_mask: [B, L_total] attention mask

        Returns:
            (y, z): updated prediction and latent embeddings
        """
        L_x = x.size(1)
        L_y = y.size(1)

        # Latent refinement iterations: update z based on [x, y, z]
        for _ in range(n):
            combined = torch.cat([x, y, z], dim=1)  # [B, L_x + L_y + L_z, D]
            hidden = self.backbone(combined, attention_mask)
            # Extract updated z
            z = hidden[:, L_x + L_y:, :]  # [B, L_z, D]

        # Output refinement: update y based on [x, y, z]
        combined = torch.cat([x, y, z], dim=1)
        hidden = self.backbone(combined, attention_mask)
        y = hidden[:, L_x:L_x + L_y, :]  # [B, L_y, D]

        return y, z

    def deep_recursion(
        self,
        x: Tensor,
        y: Tensor,
        z: Tensor,
        n: int,
        T: int,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        T-1 no_grad recursions, then 1 with grad.

        Args:
            x: [B, L_x, D] embedded prefix
            y: [B, L_y, D] prediction embeddings
            z: [B, L_z, D] latent state
            n: latent recursions per deep step
            T: number of deep recursions
            attention_mask: [B, L_total] attention mask

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

        # Compute outputs from y
        logits = self.output_head(y)  # [B, L_y, V]

        # Halt prediction based on y quality
        halt_logit = self.q_head(y)  # [B]

        # Detach for next supervision step
        return y.detach(), z.detach(), logits, halt_logit

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> dict:
        """
        Full TRM training step with deep supervision.

        Randomly splits input into prefix (x) and target (y_target),
        then iteratively refines y to predict the target continuation.

        Args:
            input_ids: [B, L] token ids
            attention_mask: [B, L] padding mask (unused, kept for API compatibility)

        Returns:
            {"loss": tensor, "accuracy": tensor, "n_sup_steps": int, "prefix_len": int}
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Randomly sample prefix length
        # Prefix must be at least min_prefix and leave room for y_len target tokens
        max_prefix = L - self.y_len
        if max_prefix < self.min_prefix:
            # Sequence too short, use minimum viable split
            prefix_len = max(1, L - self.y_len)
        else:
            prefix_len = torch.randint(self.min_prefix, max_prefix + 1, (1,)).item()

        # Split into prefix and target
        prefix_ids = input_ids[:, :prefix_len]  # [B, prefix_len]
        target_ids = input_ids[:, prefix_len:prefix_len + self.y_len]  # [B, y_len]

        # Embed prefix with positional encoding
        x = self.embeddings(prefix_ids)  # [B, prefix_len, D]

        # Initialize y and z with learned parameters + positional encoding
        y = self.y_init.expand(B, -1, -1) + self.y_pos  # [B, y_len, D]
        z = self.z_init.expand(B, -1, -1) + self.z_pos  # [B, z_len, D]

        # Create attention mask for concatenated sequence
        attn_mask = self._create_attention_mask(prefix_len, B, device)

        # Deep supervision
        total_loss = 0.0
        total_accuracy = 0.0
        actual_steps = 0

        for sup_step in range(self.N_sup):
            y, z, logits, halt_logit = self.deep_recursion(
                x, y, z, self.n, self.T, attn_mask
            )

            # Compute loss: predict target_ids from y logits
            # logits: [B, y_len, V], target_ids: [B, y_len]
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
            )

            # Compute accuracy
            preds = logits.argmax(dim=-1)  # [B, y_len]
            accuracy = (preds == target_ids).float().mean()

            # Q-head loss: predict if predictions are accurate
            q_loss = F.binary_cross_entropy_with_logits(
                halt_logit.mean(), accuracy.detach()
            )

            step_loss = ce_loss + 0.1 * q_loss
            total_loss = total_loss + step_loss
            total_accuracy = total_accuracy + accuracy.detach()
            actual_steps += 1

            # Optional early stopping when confident
            halt_prob = torch.sigmoid(halt_logit.mean())
            if halt_prob > 0.5 and sup_step > 0:
                break

        # Average over supervision steps
        loss = total_loss / actual_steps
        avg_accuracy = total_accuracy / actual_steps

        return {
            "loss": loss,
            "accuracy": avg_accuracy,
            "n_sup_steps": actual_steps,
            "prefix_len": prefix_len,
        }

    @torch.no_grad()
    def generate(
        self,
        prefix_ids: Tensor,
        steps: int = 10,
    ) -> Tensor:
        """
        Generate continuation given a prefix.

        Args:
            prefix_ids: [B, L_prefix] prefix token ids
            steps: number of refinement steps

        Returns:
            generated: [B, y_len] predicted continuation token ids
        """
        device = prefix_ids.device
        B = prefix_ids.size(0)
        prefix_len = prefix_ids.size(1)

        # Embed prefix
        x = self.embeddings(prefix_ids)  # [B, prefix_len, D]

        # Initialize y and z
        y = self.y_init.expand(B, -1, -1) + self.y_pos
        z = self.z_init.expand(B, -1, -1) + self.z_pos

        # Create attention mask
        attn_mask = self._create_attention_mask(prefix_len, B, device)

        # Refine through steps
        for _ in range(steps):
            y, z = self.latent_recursion(x, y, z, self.n, attn_mask)

        # Get final predictions
        logits = self.output_head(y)
        tokens = logits.argmax(dim=-1)

        return tokens


if __name__ == "__main__":
    # Test TRM with prefix prediction
    import yaml

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Use smaller values for testing
    config["trm"]["n"] = 1
    config["trm"]["T"] = 1
    config["trm"]["N_sup"] = 1
    config["trm"]["y_len"] = 16
    config["trm"]["z_len"] = 8
    config["trm"]["min_prefix"] = 16

    B, L = 2, 128
    V = config["model"]["vocab_size"]

    model = TRM(config)

    # Test forward
    input_ids = torch.randint(0, V, (B, L))
    attention_mask = torch.ones(B, L)

    outputs = model(input_ids, attention_mask)
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Accuracy: {outputs['accuracy'].item():.4f}")
    print(f"N_sup steps: {outputs['n_sup_steps']}")
    print(f"Prefix length: {outputs['prefix_len']}")

    assert outputs["loss"].isfinite()
    assert 0 <= outputs["accuracy"] <= 1

    # Test backward
    outputs["loss"].backward()
    print("Backward pass successful")

    # Test generation
    prefix = torch.randint(0, V, (1, 32))
    generated = model.generate(prefix, steps=3)
    print(f"Generated shape: {generated.shape}")
    assert generated.shape == (1, config["trm"]["y_len"])

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\nTRM test passed!")
