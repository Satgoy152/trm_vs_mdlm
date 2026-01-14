"""Bidirectional GPT-2 backbone shared by TRM and MDLM."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Embeddings(nn.Module):
    """
    Token embeddings + learned positional embeddings.

    Args:
        vocab_size: Size of vocabulary
        d_model: Embedding dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids: [B, L] token ids

        Returns:
            embeddings: [B, L, D]
        """
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embed(input_ids) + self.pos_embed(positions)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block (no causal masking).

    Architecture:
        x = x + attn(norm1(x))
        x = x + ff(norm2(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for module in self.ff:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: [B, L, D] input embeddings
            attention_mask: [B, L] padding mask (1 = attend, 0 = ignore)

        Returns:
            hidden: [B, L, D]
        """
        # Convert attention_mask to key_padding_mask format (True = ignore)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)

        # Pre-norm feedforward
        x = x + self.ff(self.norm2(x))

        return x


class BidirectionalGPT2(nn.Module):
    """
    GPT-2 architecture without causal masking.

    Args:
        config: Model config dict with n_layers, n_heads, d_model, d_ff, dropout
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=model_cfg["d_model"],
                n_heads=model_cfg["n_heads"],
                d_ff=model_cfg["d_ff"],
                dropout=model_cfg["dropout"],
            )
            for _ in range(model_cfg["n_layers"])
        ])

        self.final_norm = nn.LayerNorm(model_cfg["d_model"])

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: [B, L, D] input embeddings (already embedded)
            attention_mask: [B, L] optional padding mask

        Returns:
            hidden: [B, L, D] final hidden states
        """
        for layer in self.layers:
            x = layer(x, attention_mask)

        return self.final_norm(x)


class OutputHead(nn.Module):
    """
    Project hidden states to vocab logits.
    Optionally ties weights with token embeddings.

    Args:
        d_model: Hidden dimension
        vocab_size: Vocabulary size
        tie_weights: Embedding weights to tie (optional)
    """

    def __init__(self, d_model: int, vocab_size: int, tie_weights: nn.Parameter | None = None):
        super().__init__()
        self.tie_weights = tie_weights

        if tie_weights is None:
            self.proj = nn.Linear(d_model, vocab_size, bias=False)
            nn.init.normal_(self.proj.weight, std=0.02)
        else:
            self.proj = None

    def forward(self, hidden: Tensor) -> Tensor:
        """
        Args:
            hidden: [B, L, D]

        Returns:
            logits: [B, L, V]
        """
        if self.tie_weights is not None:
            return F.linear(hidden, self.tie_weights)
        return self.proj(hidden)


class QHead(nn.Module):
    """
    Halting prediction head for TRM.
    Mean-pools sequence, projects to scalar, applies sigmoid.

    Args:
        d_model: Hidden dimension
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.proj:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, hidden: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            hidden: [B, L, D]
            attention_mask: [B, L] optional mask for pooling

        Returns:
            halt_prob: [B]
        """
        if attention_mask is not None:
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)

        return torch.sigmoid(self.proj(pooled).squeeze(-1))


if __name__ == "__main__":
    # Test all model components
    import yaml

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    B, L = 2, 512
    D = config["model"]["d_model"]
    V = config["model"]["vocab_size"]

    # Test Embeddings
    embeddings = Embeddings(V, D, config["model"]["max_seq_len"], config["model"]["dropout"])
    input_ids = torch.randint(0, V, (B, L))
    emb_out = embeddings(input_ids)
    print(f"Embeddings: input {input_ids.shape} -> output {emb_out.shape}")
    assert emb_out.shape == (B, L, D)

    # Test BidirectionalGPT2
    backbone = BidirectionalGPT2(config)
    attention_mask = torch.ones(B, L)
    hidden = backbone(emb_out, attention_mask)
    print(f"Backbone: input {emb_out.shape} -> output {hidden.shape}")
    assert hidden.shape == (B, L, D)

    # Test OutputHead
    output_head = OutputHead(D, V)
    logits = output_head(hidden)
    print(f"OutputHead: input {hidden.shape} -> output {logits.shape}")
    assert logits.shape == (B, L, V)

    # Test OutputHead with tied weights
    output_head_tied = OutputHead(D, V, tie_weights=embeddings.token_embed.weight)
    logits_tied = output_head_tied(hidden)
    print(f"OutputHead (tied): input {hidden.shape} -> output {logits_tied.shape}")
    assert logits_tied.shape == (B, L, V)

    # Test QHead
    q_head = QHead(D)
    halt_prob = q_head(hidden, attention_mask)
    print(f"QHead: input {hidden.shape} -> output {halt_prob.shape}")
    assert halt_prob.shape == (B,)
    assert (halt_prob >= 0).all() and (halt_prob <= 1).all()

    # Count parameters
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"\nBackbone parameters: {total_params:,}")

    print("\nModel test passed!")
