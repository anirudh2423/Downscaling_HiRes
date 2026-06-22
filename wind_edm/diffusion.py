"""
diffusion.py — EDM preconditioning and training loss (Karras et al. 2022).
"""

import torch


class EDM:
    """
    EDM preconditioning — https://arxiv.org/abs/2206.00364.
    Works for any number of target channels.
    """

    def __init__(
        self,
        sigma_max  = 80.0,
        sigma_min  = 0.02,
        sigma_data = 0.5,
        P_mean     = -1.2,
        P_std      = 1.2,
        rho        = 7.0,
    ):
        self.sigma_min  = sigma_min
        self.sigma_max  = sigma_max
        self.sigma_data = sigma_data
        self.P_mean     = P_mean
        self.P_std      = P_std
        self.rho        = rho

    def c_skip(self, sigma):
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    def c_in(self, sigma):
        return 1.0 / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    def c_noise(self, sigma):
        return sigma.log() / 4.0

    def loss(self, model, x0, cond):
        """
        x0   : (B, 2, H, W)  clean normalised target (UGRD, VGRD)
        cond : (B, 3, H, W)  conditioning (topo, SVF, CSZA)
        """
        B      = x0.shape[0]
        device = x0.device
        log_sigma = torch.randn(B, device=device) * self.P_std + self.P_mean
        sigma     = log_sigma.exp().view(B, 1, 1, 1)
        eps       = torch.randn_like(x0)
        x_noisy   = x0 + sigma * eps

        x_in         = torch.cat([self.c_in(sigma) * x_noisy, cond], dim=1)
        noise_labels = self.c_noise(sigma.view(B))
        F_x  = model(x_in, noise_labels).sample           # UNet2DModel returns named tuple
        D_x  = self.c_skip(sigma) * x_noisy + self.c_out(sigma) * F_x

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        return (weight * (D_x - x0) ** 2).mean()

    def sample_sigmas(self, n_steps):
        rho  = self.rho
        step = torch.arange(n_steps + 1)
        return (
            self.sigma_max ** (1 / rho)
            + step / n_steps * (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))
        ) ** rho

    @torch.no_grad()
    def sample(self, model, cond, n_steps=18, n_target_channels=2, device="cuda"):
        """
        Euler-Maruyama sampler (Algorithm 2, Karras et al. 2022).
        cond : (B, 3, H, W)
        Returns (B, n_target_channels, H, W) denoised output.
        """
        B, _, H, W = cond.shape
        sigmas = self.sample_sigmas(n_steps).to(device)

        x = torch.randn(B, n_target_channels, H, W, device=device) * sigmas[0]

        for i in range(n_steps):
            sigma      = sigmas[i]
            sigma_next = sigmas[i + 1]

            x_in         = torch.cat([self.c_in(sigma) * x, cond], dim=1)
            noise_labels = self.c_noise(torch.full((B,), sigma, device=device))

            F_x = model(x_in, noise_labels).sample        # UNet2DModel returns named tuple
            D_x = self.c_skip(sigma) * x + self.c_out(sigma) * F_x

            d = (x - D_x) / sigma
            x = x + (sigma_next - sigma) * d

        return x
