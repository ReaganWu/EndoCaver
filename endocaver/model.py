"""EndoCaver model definition.

This file contains the public reference implementation of the architecture used
in the paper.  The forward pass returns two tensors:

    restored_rgb, segmentation_probability = model(x)

Both outputs are in [0, 1].  Input images should be resized to 224 x 224 and
scaled to [0, 1].
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath
from transformers import AutoModelForImageClassification


def conv_1x1_bn(inp: int, oup: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU(),
    )


def conv_nxn_bn(inp: int, oup: int, kernel_size: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU(),
    )


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b p n (h d) -> b p h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        out = rearrange(out, "b p h n d -> b p n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([Attention(dim, heads, dim_head, dropout), FeedForward(dim, mlp_dim, dropout)])
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class CrossAttentionBlock(nn.Module):
    """Deblurring-Segmentation Aligner (DSA)."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, seg_feat: torch.Tensor, deblur_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = seg_feat.shape
        q = seg_feat.flatten(2).transpose(1, 2)
        kv = deblur_feat.flatten(2).transpose(1, 2)
        out, _ = self.attn(q, kv, kv)
        out = self.proj(out).transpose(1, 2).reshape(b, c, h, w)
        return seg_feat + out


class UpMV2Block(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, stride: int = 1, expansion: int = 4, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(in_chans * expansion)
        if in_chans == out_chans:
            self.conv = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, out_chans, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_chans),
            )
            self.use_res_connect = True
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(out_chans),
                nn.SiLU(),
            )
            self.use_res_connect = False
        self.drop = DropPath(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_res_connect:
            out = self.drop(out) + x
        return out


class MobileViTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        channel: int,
        kernel_size: int,
        patch_size: Tuple[int, int],
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ph, self.pw = patch_size
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        x = self.conv1(x)
        x = self.conv2(x)
        _, _, h, w = x.shape
        x = rearrange(x, "b d (h ph) (w pw) -> b (ph pw) (h w) d", ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, "b (ph pw) (h w) d -> b d (h ph) (w pw)", h=h // self.ph, w=w // self.pw, ph=self.ph, pw=self.pw)
        x = self.conv3(x)
        x = torch.cat((x, y), dim=1)
        return self.conv4(x)


class GlobalAttentionModule(nn.Module):
    """Global Attention Module (GAM) for cross-scale aggregation."""

    def __init__(self, in_channels_list: Sequence[int], embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.proj_layers = nn.ModuleList([nn.Conv2d(c, embed_dim, kernel_size=1) for c in in_channels_list])
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.unified_layers = nn.ModuleList([nn.Conv2d(embed_dim, c, kernel_size=1) for c in in_channels_list])

    def forward(self, feats: List[torch.Tensor]):
        target_size = feats[0].shape[2:]
        proj_feats = [proj(F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)) for f, proj in zip(feats, self.proj_layers)]
        x = torch.stack(proj_feats, dim=1).mean(1)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        attn_out, attn_weights = self.attn(tokens, tokens, tokens)
        attn_out = self.norm(attn_out + tokens)
        attn_feat = attn_out.transpose(1, 2).reshape(b, c, h, w)
        modulated = []
        for f, unify in zip(feats, self.unified_layers):
            f_up = F.interpolate(attn_feat, size=f.shape[2:], mode="bilinear", align_corners=False)
            modulated.append(f * unify(f_up))
        return attn_feat, modulated, attn_weights


class DeblurDecoder(nn.Module):
    def __init__(self, image_size=(224, 224), dims=(64, 80, 96), channels=(32, 64, 160, 256), activation="sigmoid"):
        super().__init__()
        self.stem = nn.ModuleList([
            nn.ModuleList([UpMV2Block(channels[3], channels[2]), MobileViTBlock(dims[2], 2, channels[2], 3, (2, 2), dims[2] * 4)]),
            nn.ModuleList([UpMV2Block(channels[2], channels[1]), MobileViTBlock(dims[1], 4, channels[1], 3, (4, 4), dims[1] * 4)]),
            nn.ModuleList([UpMV2Block(channels[1], channels[0]), MobileViTBlock(dims[0], 3, channels[0], 3, (8, 8), dims[0] * 4)]),
        ])
        self.trunk = nn.ModuleList([
            nn.ModuleList([UpMV2Block(channels[0], channels[0]), UpMV2Block(channels[0], channels[0] // 2)]),
            nn.ModuleList([UpMV2Block(channels[0] // 2, channels[0] // 2), UpMV2Block(channels[0] // 2, channels[0] // 4)]),
        ])
        self.final_conv = nn.Conv2d(channels[0] // 4 + 3, 3, 1)
        self.activation = activation

    def forward(self, x_list: List[torch.Tensor], x_input: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        x = x_list.pop()
        for enhance, upconv in self.stem:
            skip = x_list.pop()
            x = enhance(x)
            x = upconv(x)
            x = x + skip
            outputs.append(x)
        for mobvitatt, upsample in self.trunk:
            x = mobvitatt(x)
            x = upsample(x)
            outputs.append(x)
        x = torch.cat([x, x_input], dim=1)
        x = self.final_conv(x)
        if self.activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.activation == "relu":
            x = F.relu(x)
        outputs.append(x)
        return outputs


class SegmentationDecoder(nn.Module):
    def __init__(self, image_size=(224, 224), dims=(64, 80, 96), channels=(32, 64, 160, 256), activation="sigmoid"):
        super().__init__()
        self.stem = nn.ModuleList([
            nn.ModuleList([CrossAttentionBlock(channels[3]), UpMV2Block(channels[3], channels[2]), MobileViTBlock(dims[2], 3, channels[2], 3, (2, 2), dims[2] * 4), CrossAttentionBlock(channels[2])]),
            nn.ModuleList([CrossAttentionBlock(channels[2]), UpMV2Block(channels[2], channels[1]), MobileViTBlock(dims[1], 4, channels[1], 3, (4, 4), dims[1] * 4), CrossAttentionBlock(channels[1])]),
            nn.ModuleList([CrossAttentionBlock(channels[1]), UpMV2Block(channels[1], channels[0]), MobileViTBlock(dims[0], 2, channels[0], 3, (8, 8), dims[0] * 4), CrossAttentionBlock(channels[0])]),
        ])
        # Keep the original checkpoint key spelling (attention_experssion) for
        # compatibility with the released arXiv/ICASSP weights.
        self.attention_experssion = nn.Sequential(conv_nxn_bn(128, channels[0], 3, 1), UpMV2Block(channels[0], channels[0]))
        self.cross_attn_future = CrossAttentionBlock(channels[0])
        self.trunk = nn.ModuleList([
            nn.ModuleList([UpMV2Block(channels[0], channels[0]), UpMV2Block(channels[0], channels[0] // 2)]),
            nn.ModuleList([UpMV2Block(channels[0] // 2, channels[0] // 2), UpMV2Block(channels[0] // 2, channels[0] // 4)]),
        ])
        self.final_conv = nn.Conv2d(channels[0] // 4, 1, 1)
        self.activation = activation

    def forward(self, x_list: List[torch.Tensor], encoder_enhanced: List[torch.Tensor], deblur_outputs: List[torch.Tensor], attention_feature: torch.Tensor) -> torch.Tensor:
        x = x_list.pop()
        deblur_idx = 0
        for pre_cross_attn, enhance, mobvit, cross_attn in self.stem:
            skip = x_list.pop()
            e = encoder_enhanced.pop()
            x = pre_cross_attn(x, e)
            deblur_feat = deblur_outputs[deblur_idx]
            deblur_idx += 1
            x = enhance(x)
            x = mobvit(x)
            x = x + skip
            x = cross_attn(x, deblur_feat)
        attention_feature = self.attention_experssion(attention_feature)
        x = self.cross_attn_future(x, attention_feature)
        for mobvitatt, upsample in self.trunk:
            x = mobvitatt(x)
            x = upsample(x)
        x = self.final_conv(x)
        if self.activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.activation == "relu":
            x = F.relu(x)
        return x


class EndoCaver(nn.Module):
    """EndoCaver: a unidirectional-guided dual-decoder transformer.

    The public class name follows the paper title, while the main module names
    keep checkpoint compatibility with the original ICASSP/arXiv experiments:
    encoder + global_attention_fusion (GAM) + deblurring decoder + segmentation
    decoder with DSA blocks.
    """

    def __init__(self, encoder_name: str = "nvidia/mit-b0", local_files_only: bool = False):
        super().__init__()
        self.encoder = AutoModelForImageClassification.from_pretrained(encoder_name, local_files_only=local_files_only)
        self.global_attention_fusion = GlobalAttentionModule([32, 64, 160, 256], embed_dim=128, num_heads=4)
        self.decoder_deblur = DeblurDecoder(channels=[32, 64, 160, 256], activation="sigmoid")
        self.decoder_seg = SegmentationDecoder(channels=[32, 64, 160, 256], activation="sigmoid")

    def forward(self, x: torch.Tensor):
        encoder_output = self.encoder(x, output_hidden_states=True)
        hidden_states = list(encoder_output.hidden_states)
        attn_feature, modulated_feature, _ = self.global_attention_fusion(hidden_states[:])
        deblur_outputs = self.decoder_deblur(modulated_feature[:], x)
        seg_output = self.decoder_seg(hidden_states[:], modulated_feature[:], deblur_outputs, attn_feature)
        return deblur_outputs[-1], seg_output


# Paper-style alias for users who want the architecture name explicitly.
UnidirectionalGuidedDualDecoder = EndoCaver

# Backward-compatible alias used by early experiment scripts and checkpoints.
Endocaver_v4 = EndoCaver


if __name__ == "__main__":
    model = EndoCaver()
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        restored, mask = model(x)
    print("restored:", tuple(restored.shape), "mask:", tuple(mask.shape))
