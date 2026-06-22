"""
EDM UNet for AORC 2 m temperature downscaling.

Input:
    cat([c_in(sigma) * noisy_temperature, cond], dim=1)

Channels:
    noisy target: 1 channel, normalized AORC TMP_2maboveground
    cond: 4 channels, topo, SVF, CSZA, ERA5 2m_temperature
    output: 1 channel, denoised normalized AORC temperature
"""

from diffusers import UNet2DModel

from config import CFG


def build_model():
    return UNet2DModel(
        sample_size=CFG.patch_size,
        in_channels=CFG.input_channels,
        out_channels=CFG.target_channels,
        layers_per_block=CFG.num_res_blocks,
        block_out_channels=tuple(CFG.base_ch * m for m in CFG.ch_mult),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


if __name__ == "__main__":
    import torch

    model = build_model()
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total:.1f}M")

    x_noisy = torch.randn(2, CFG.target_channels, CFG.patch_size, CFG.patch_size)
    cond = torch.randn(2, CFG.cond_channels, CFG.patch_size, CFG.patch_size)
    x_input = torch.cat([x_noisy, cond], dim=1)
    noise_labels = torch.rand(2) * 10
    out = model(x_input, noise_labels).sample
    print(f"Input: {x_input.shape} -> Output: {out.shape}")
