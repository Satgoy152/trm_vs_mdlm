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

    def _create_remasked_input(self, masked_input: Tensor, logits: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Create input for step 2 by keeping some predictions and remasking others.

        Args:
            masked_input: [B, L] input from step 1 (with MASK tokens)
            logits: [B, L, V] predictions from step 1
            mask: [B, L] boolean mask of originally masked positions

        Returns:
            new_input_ids: [B, L] input for step 2
            remaining_mask: [B, L] mask of tokens that are still MASK in step 2
        """
        # Get predictions
        preds = logits.argmax(dim=-1)

        # Decide which to keep (50% random)
        # We fill if it was originally masked AND random < 0.5
        # Otherwise it stays as MASK token (from masked_input)
        keep_prob = 0.5
        rand = torch.rand(mask.shape, device=mask.device)
        fill_mask = mask & (rand < keep_prob)
        
        remaining_mask = mask & (~fill_mask)

        new_input_ids = masked_input.clone()
        new_input_ids[fill_mask] = preds[fill_mask]

        return new_input_ids, remaining_mask

    def _compute_entropy_stats(self, logits: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute mean entropy for masked and unmasked positions.
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1) # [B, L]
        
        # Avoid NaN if mask is empty or full
        if mask.any():
            mean_masked = entropy[mask].mean()
        else:
            mean_masked = torch.tensor(0.0, device=logits.device)
            
        if (~mask).any():
            mean_unmasked = entropy[~mask].mean()
        else:
            mean_unmasked = torch.tensor(0.0, device=logits.device)
            
        return mean_masked, mean_unmasked

    def _compute_loss(self, logits: Tensor, targets: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute loss on originally masked positions.
        """
        masked_logits = logits[mask]
        masked_targets = targets[mask]

        if masked_logits.numel() == 0:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            accuracy = torch.tensor(1.0, device=logits.device)
        else:
            loss = F.cross_entropy(masked_logits, masked_targets)
            preds = masked_logits.argmax(dim=-1)
            accuracy = (preds == masked_targets).float().mean()

        return loss, accuracy

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> dict:
        """
        Two-step training forward pass.

        1. Sample mask and create x_t1
        2. Step 1 (no_grad): Predict -> x_t2 (keep/remask)
        3. Step 2 (grad): Predict
        4. Loss on originally masked tokens

        Args:
            input_ids: [B, L] token ids
            attention_mask: [B, L] padding mask

        Returns:
            {"loss": tensor, "accuracy": tensor, "mask_ratio": float}
        """
        B, L = input_ids.shape
        device = input_ids.device

        # 1. Initial Masking
        mask_ratio = torch.empty(1).uniform_(self.mask_ratio_min, self.mask_ratio_max).item()
        
        # Create mask (True = masked)
        rand = torch.rand(B, L, device=device)
        mask = (rand < mask_ratio) & (attention_mask == 1)

        # Store original tokens (ground truth)
        targets = input_ids.clone()

        # Apply initial mask
        masked_input = input_ids.clone()
        masked_input[mask] = self.mask_token_id

        # 2. Step 1: Initial prediction 
        x_1 = self.embeddings(masked_input)
        hidden_1 = self.backbone(x_1, attention_mask)
        logits_1 = self.output_head(hidden_1)

        # 2.1 Get loss (standard MDLM loss)
        loss_1, accuracy_1 = self._compute_loss(logits_1, targets, mask)
        
        # 2.2 Compute entropy stats for Step 1
        ent_masked_1, ent_unmasked_1 = self._compute_entropy_stats(logits_1, mask)
        
        # # Create input for step 2
        # x_t2_input_ids, mask_2 = self._create_remasked_input(masked_input, logits_1.detach(), mask)

        # # 3. Step 2: Refinement pass (with gradients)
        # x_2 = self.embeddings(x_t2_input_ids)
        # hidden_2 = self.backbone(x_2, attention_mask)
        # logits_2 = self.output_head(hidden_2)

        # # 4. Loss calculation (on originally masked positions)
        # loss_2, accuracy_2 = self._compute_loss(logits_2, targets, mask)

        # # 4.1 Compute entropy stats for Step 2
        # ent_masked_2, ent_unmasked_2 = self._compute_entropy_stats(logits_2, mask_2)

        # loss = (loss_1 + loss_2) / 2
        # accuracy = (accuracy_1 + accuracy_2) / 2
        loss = loss_1
        accuracy = accuracy_1

        ent_masked_2 = ent_masked_1
        ent_unmasked_2 = ent_unmasked_1


        return {
            "loss": loss,
            "accuracy": accuracy_1,
            "mask_ratio": mask_ratio,
            "ent_masked_1": ent_masked_1,
            "ent_unmasked_1": ent_unmasked_1,
            "ent_masked_2": ent_masked_2,
            "ent_unmasked_2": ent_unmasked_2,
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
