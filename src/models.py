"""Model architectures for the Vision Transformer assignment (Part 2).

Two models, trained from random initialization on CIFAR-100:

  1. SwinTransformer  -- the primary model. Implements patch embedding,
     windowed multi-head self-attention, shifted-window attention,
     hierarchical stages with patch merging, and relative position bias.
  2. VisionTransformer -- the plain ViT baseline. Implements patch
     embedding, a class token, learnable absolute positional embeddings,
     and standard global multi-head self-attention.

Both models expose every setting the assignment requires as configurable
constructor arguments (see SwinConfig / ViTConfig below), so a single
hyperparameter search can produce a ViT baseline whose trainable parameter
count is within 10% of the Swin model's, as required by Part 2.2.

Design note on why relative position bias is included: Swin does not use
absolute positional embeddings the way plain ViT does. Instead, each
windowed-attention operation learns a relative position bias indexed by the
relative offset between the query and key patch within a window. This is a
real, non-optional source of trainable parameters for Swin (see Part 4.2,
which explicitly lists "Relative-position parameters" as a component of the
Swin parameter count), so it is implemented here rather than omitted.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class DropPath(nn.Module):
    """Stochastic depth: randomly drops entire residual branches per-sample.

    Implements the "stochastic-depth rate" setting required for the Swin
    model. At drop_prob=0 this is a no-op (identity), which is how it's used
    by default in the plain ViT baseline.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Per-sample binary mask, broadcast over all non-batch dimensions.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    """Standard Transformer feed-forward block: D -> hidden -> D."""

    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """Conv-based patch embedding: splits the image into non-overlapping
    patches and linearly projects each to `embed_dim`, in one strided conv.
    """

    def __init__(self, input_resolution: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        if input_resolution % patch_size != 0:
            raise ValueError(
                f"input_resolution ({input_resolution}) must be divisible by "
                f"patch_size ({patch_size})"
            )
        self.grid_size = input_resolution // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # x: (B, C, H, W) -> (B, embed_dim, grid, grid) -> (B, grid*grid, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameter count, matching the assignment's required
    `sum(p.numel() for p in model.parameters() if p.requires_grad)` check.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Swin Transformer (primary model)
# ---------------------------------------------------------------------------


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(batch, height, width, channels) -> (num_windows*batch, ws, ws, channels)."""
    batch_size, height, width, channels = x.shape
    x = x.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, channels
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)
    return windows


def window_reverse(
    windows: torch.Tensor, window_size: int, height: int, width: int
) -> torch.Tensor:
    """Inverse of window_partition: (windows*batch, ws, ws, ch) -> (batch, height, width, ch)."""
    batch_size = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(
        batch_size, height // window_size, width // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, -1)
    return x


class WindowAttention(nn.Module):
    """Multi-head self-attention restricted to non-overlapping windows, with
    a learned relative position bias per head.

    The relative position bias table has shape
    ((2*window_size - 1)^2, num_heads): one learned scalar bias per head for
    every possible relative (dy, dx) offset between two patches inside a
    window. This is looked up per query-key pair via `relative_position_index`.
    """

    def __init__(self, dim: int, window_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if head_dim * num_heads != dim:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Precompute the (window_size^2, window_size^2) index into the bias
        # table for every pair of positions within a window. Fixed given
        # window_size, so it's a buffer, not a parameter.
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, ws, ws)
        coords_flatten = torch.flatten(coords, 1)  # (2, ws*ws)
        # (2, ws*ws, ws*ws)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (ws*ws, ws*ws, 2)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (ws*ws, ws*ws)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass."""
        # x: (num_windows*batch, num_tokens, channels) where num_tokens = window_size**2
        windowed_batch, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            windowed_batch, num_tokens, 3, self.num_heads, channels // self.num_heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # (windowed_batch, heads, num_tokens, num_tokens)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(num_tokens, num_tokens, -1)
        # (heads, num_tokens, num_tokens)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: (num_windows, num_tokens, num_tokens); attn currently
            # (windowed_batch, heads, num_tokens, num_tokens) where
            # windowed_batch = batch * num_windows. Reshape to apply mask per-window.
            num_windows = mask.shape[0]
            attn = attn.view(
                windowed_batch // num_windows, num_windows, self.num_heads, num_tokens, num_tokens
            )
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, num_tokens, num_tokens)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(windowed_batch, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """One Swin block: (shifted) window attention + MLP, both pre-norm with
    residual connections and stochastic depth.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        # If the feature map is no larger than the window, there is nothing
        # to shift: use the whole map as a single window with no shift.
        if input_resolution <= window_size:
            shift_size = 0
            window_size = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads, dropout=dropout
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), dropout=dropout)

        if self.shift_size > 0:
            self.register_buffer("attn_mask", self._build_shift_mask(), persistent=False)
        else:
            self.attn_mask = None

    def _build_shift_mask(self) -> torch.Tensor:
        """Precompute the attention mask that prevents shifted windows from
        attending across the cyclic-shift wrap boundary (standard Swin trick:
        assign each region of the shifted image a group id, then mask out
        attention between tokens from different groups within a window).
        """
        height = width = self.input_resolution
        img_mask = torch.zeros((1, height, width, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = count
                count += 1

        mask_windows = window_partition(img_mask, self.window_size)  # (num_windows, ws, ws, 1)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        height = width = self.input_resolution
        batch_size, seq_len, channels = x.shape
        assert seq_len == height * width, (
            f"input feature has wrong size: {seq_len} != {height}*{width}"
        )

        shortcut = x
        x = self.norm1(x)
        x = x.view(batch_size, height, width, channels)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)  # (nW*batch, ws, ws, channels)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channels)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, channels)
        shifted_x = window_reverse(attn_windows, self.window_size, height, width)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(batch_size, height * width, channels)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Downsamples a (H, W) feature grid to (H/2, W/2) and doubles channels:
    concatenate each 2x2 neighborhood (4C), normalize, then linearly project
    down to 2C. This is the mechanism that gives Swin its hierarchy.
    """

    def __init__(self, input_resolution: int, dim: int):
        super().__init__()
        if input_resolution % 2 != 0:
            raise ValueError(f"PatchMerging requires an even resolution, got {input_resolution}")
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        height = width = self.input_resolution
        batch_size, seq_len, channels = x.shape
        assert seq_len == height * width, "input feature has wrong size"

        x = x.view(batch_size, height, width, channels)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (batch, height/2, width/2, 4*channels)
        x = x.view(batch_size, -1, 4 * channels)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinStage(nn.Module):
    """One hierarchical stage: a sequence of Swin blocks (alternating regular
    and shifted-window attention), optionally followed by patch merging.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: list[float],
        downsample: bool,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drop_path[i],
                )
                for i in range(depth)
            ]
        )
        self.downsample = PatchMerging(input_resolution, dim) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


@dataclasses.dataclass
class SwinConfig:
    """All Swin settings the assignment requires to be configurable."""

    input_resolution: int = 32
    patch_size: int = 4
    in_channels: int = 3
    window_size: int = 4
    embed_dim: int = 96
    num_stages: int = 3
    depths: tuple[int, ...] = (2, 2, 6)
    num_heads: tuple[int, ...] = (3, 6, 12)
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path_rate: float = 0.1
    num_classes: int = 100

    def __post_init__(self):
        if len(self.depths) != self.num_stages:
            raise ValueError(
                f"len(depths)={len(self.depths)} must equal num_stages={self.num_stages}"
            )
        if len(self.num_heads) != self.num_stages:
            raise ValueError(
                f"len(num_heads)={len(self.num_heads)} must equal num_stages={self.num_stages}"
            )


class SwinTransformer(nn.Module):
    """Hierarchical Swin Transformer for CIFAR-100 classification."""

    def __init__(self, config: SwinConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbed(
            input_resolution=config.input_resolution,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )
        self.pos_drop = nn.Dropout(config.dropout)

        grid_size = config.input_resolution // config.patch_size
        total_blocks = sum(config.depths)
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        dim = config.embed_dim
        resolution = grid_size
        block_idx = 0
        for stage_idx in range(config.num_stages):
            depth = config.depths[stage_idx]
            heads = config.num_heads[stage_idx]
            is_last_stage = stage_idx == config.num_stages - 1
            # Window size can't exceed the current feature-map resolution.
            effective_window = min(config.window_size, resolution)
            stage = SwinStage(
                dim=dim,
                input_resolution=resolution,
                depth=depth,
                num_heads=heads,
                window_size=effective_window,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                drop_path=dpr[block_idx : block_idx + depth],
                downsample=(not is_last_stage) and resolution % 2 == 0,
            )
            self.stages.append(stage)
            block_idx += depth
            if (not is_last_stage) and resolution % 2 == 0:
                dim *= 2
                resolution //= 2

        self.final_dim = dim
        self.norm = nn.LayerNorm(self.final_dim)
        self.head = nn.Linear(self.final_dim, config.num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # global average pool over remaining tokens
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# Plain Vision Transformer (baseline)
# ---------------------------------------------------------------------------


class MultiHeadSelfAttention(nn.Module):
    """Standard *global* multi-head self-attention (no windowing) used by
    the plain ViT baseline.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        head_dim = dim // num_heads
        if head_dim * num_heads != dim:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, channels // self.num_heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)  # every token attends to every other token
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ViTBlock(nn.Module):
    """Standard pre-norm Transformer encoder block: global MHSA + MLP."""

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float, dropout: float, drop_path: float
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, dropout=dropout)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@dataclasses.dataclass
class ViTConfig:
    """All settings the assignment requires to be configurable for the
    plain ViT baseline. Kept structurally parallel to SwinConfig so the two
    models are easy to compare and to parameter-match.
    """

    input_resolution: int = 32
    patch_size: int = 4
    in_channels: int = 3
    embed_dim: int = 288
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path_rate: float = 0.0
    num_classes: int = 100


class VisionTransformer(nn.Module):
    """Plain, single-scale Vision Transformer with global self-attention."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbed(
            input_resolution=config.input_resolution,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        self.pos_drop = nn.Dropout(config.dropout)

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    drop_path=dpr[i],
                )
                for i in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.patch_embed(x)  # (batch, num_patches, channels)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_output = x[:, 0]  # classify using the class token, not average pooling
        return self.head(cls_output)


if __name__ == "__main__":
    swin = SwinTransformer(SwinConfig())
    vit = VisionTransformer(ViTConfig())

    swin_params = count_parameters(swin)
    vit_params = count_parameters(vit)
    pct_diff = abs(swin_params - vit_params) / swin_params * 100

    print(f"Swin params: {swin_params:,}")
    print(f"ViT params:  {vit_params:,}")
    print(f"Percent difference: {pct_diff:.2f}% (must be <= 10%)")

    dummy = torch.randn(2, 3, 32, 32)
    print("Swin output shape:", swin(dummy).shape)
    print("ViT output shape:", vit(dummy).shape)
