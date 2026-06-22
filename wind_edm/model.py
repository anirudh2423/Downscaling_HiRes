"""
model.py — EDM UNet using HuggingFace diffusers UNet2DModel.

Input tensor fed to forward():
    (B, 7, H, W)  — cat([c_in(sigma)*x_noisy, cond], dim=1)
    where x_noisy : (B, 2, H, W)  noisy UGRD + VGRD
          cond    : (B, 5, H, W)  topo, SVF, CSZA, coarse U, coarse V

Channel progression (base_ch=64, ch_mult=(1,2,4,8), attn at 32×32 and 16×16):
    256×256 →  64ch  (no attn)
    128×128 → 128ch  (no attn)
     64× 64 → 256ch  (no attn)
     32× 32 → 512ch  (attn)
     16× 16 → 512ch  (attn, mid-block)
    (symmetric decoder)
    256×256 →   2ch  output
"""

from diffusers import UNet2DModel
from config import CFG


def build_model():
    model = UNet2DModel(
        sample_size     = CFG.patch_size,
        in_channels     = CFG.input_channels,
        out_channels    = 2,              # UGRD, VGRD
        layers_per_block= CFG.num_res_blocks,
        block_out_channels = tuple(CFG.base_ch * m for m in CFG.ch_mult),  # (64,128,256,512)
        down_block_types=(
            "DownBlock2D",      # 256×256
            "DownBlock2D",      # 128×128
            "DownBlock2D",      # 64×64
            "AttnDownBlock2D",  # 32×32  ← attention
        ),
        up_block_types=(
            "AttnUpBlock2D",    # 32×32  ← attention
            "UpBlock2D",        # 64×64
            "UpBlock2D",        # 128×128
            "UpBlock2D",        # 256×256
        ),
        # mid-block always has attention → covers 16×16
    )
    return model


if __name__ == "__main__":
    import torch
    model = build_model()
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total:.1f}M")

    x_noisy = torch.randn(2, 2, 256, 256)
    cond    = torch.randn(2, CFG.cond_channels, 256, 256)
    x_input = torch.cat([x_noisy, cond], dim=1)   # (2, 7, 256, 256)

    noise_labels = torch.rand(2) * 10
    out = model(x_input, noise_labels).sample
    print(f"Input: {x_input.shape} → Output: {out.shape}")   # expect (2, 2, 256, 256)
