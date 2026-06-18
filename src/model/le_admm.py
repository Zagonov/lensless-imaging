import torch
from torch import nn

from src.model.admm_utils import (
    crop_center,
    diff_x,
    diff_y,
    get_fft_size,
    make_diff_otfs,
    pad_to_shape,
    psf_to_fourier,
    shrink,
)


class LeADMM(nn.Module):
    def __init__(
        self,
        n_iter,
        mu1_init=1e-4,
        mu2_init=1e-4,
        mu3_init=1e-4,
        tau_init=2e-4,
        learnable=True,
        padded_scale=2,
        roi_top=80,
        roi_left=100,
        roi_height=200,
        roi_width=266,
        eps=1e-8,
        return_full=True,
    ):
        super().__init__()
        self.n_iter = n_iter
        self.learnable = learnable
        self.padded_scale = padded_scale
        self.roi_top = roi_top
        self.roi_left = roi_left
        self.roi_height = roi_height
        self.roi_width = roi_width
        self.eps = eps
        self.return_full = return_full

        if self.learnable:
            self.mu1 = nn.Parameter(torch.full((n_iter,), mu1_init**0.5))
            self.mu2 = nn.Parameter(torch.full((n_iter,), mu2_init**0.5))
            self.mu3 = nn.Parameter(torch.full((n_iter,), mu3_init**0.5))
            self.tau = nn.Parameter(torch.full((n_iter,), tau_init**0.5))
        else:
            self.register_buffer("mu1", torch.full((n_iter,), mu1_init))
            self.register_buffer("mu2", torch.full((n_iter,), mu2_init))
            self.register_buffer("mu3", torch.full((n_iter,), mu3_init))
            self.register_buffer("tau", torch.full((n_iter,), tau_init))

    def forward(self, lensless, psf, **batch):
        sensor_height, sensor_width = lensless.shape[-2:]
        device = lensless.device
        full_height = get_fft_size(sensor_height, self.padded_scale)
        full_width = get_fft_size(sensor_width, self.padded_scale)

        m_full = pad_to_shape(lensless, full_height, full_width)
        crop_mask = pad_to_shape(torch.ones_like(lensless), full_height, full_width)
        psf_full = pad_to_shape(psf, full_height, full_width)

        hf = psf_to_fourier(psf_full)
        hf_conj = torch.conj(hf)
        hf_sqaured = hf.abs().square()

        dxf, dyf = make_diff_otfs(full_height, full_width, device, lensless.dtype)
        dxf_conj = torch.conj(dxf)
        dyf_conj = torch.conj(dyf)
        dx_power = dxf.abs().square()
        dy_power = dyf.abs().square()

        x = torch.zeros_like(m_full)
        u_x = torch.zeros_like(m_full)
        u_y = torch.zeros_like(m_full)
        v = torch.zeros_like(m_full)
        w = torch.zeros_like(m_full)
        alpha1 = torch.zeros_like(m_full)
        alpha2_x = torch.zeros_like(m_full)
        alpha2_y = torch.zeros_like(m_full)
        alpha3 = torch.zeros_like(m_full)

        if self.learnable:
            mu1 = self.mu1.square() + self.eps
            mu2 = self.mu2.square() + self.eps
            mu3 = self.mu3.square() + self.eps
            tau = self.tau.square() + self.eps
        else:
            mu1 = self.mu1
            mu2 = self.mu2
            mu3 = self.mu3
            tau = self.tau

        for i in range(self.n_iter):
            mu1_i = mu1[i].to(device=device, dtype=lensless.dtype)
            mu2_i = mu2[i].to(device=device, dtype=lensless.dtype)
            mu3_i = mu3[i].to(device=device, dtype=lensless.dtype)
            tau_i = tau[i].to(device=device, dtype=lensless.dtype)

            dx = diff_x(x)
            dy = diff_y(x)
            threshold = tau_i / mu2_i
            u_x = shrink(dx + alpha2_x / mu2_i, threshold)
            u_y = shrink(dy + alpha2_y / mu2_i, threshold)

            x_fft = torch.fft.fft2(x)
            hx = torch.fft.ifft2(hf * x_fft).real
            v = (alpha1 + mu1_i * hx + m_full) / (crop_mask + mu1_i)
            w = torch.clamp(alpha3 / mu3_i + x, min=0.0)

            ht_v_term = hf_conj * torch.fft.fft2(mu1_i * v - alpha1)
            dx_term = dxf_conj * torch.fft.fft2(mu2_i * u_x - alpha2_x)
            dy_term = dyf_conj * torch.fft.fft2(mu2_i * u_y - alpha2_y)
            w_term = torch.fft.fft2(mu3_i * w - alpha3)
            denom = mu1_i * hf_sqaured + mu2_i * dx_power + mu2_i * dy_power + mu3_i

            rhs = ht_v_term + dx_term + dy_term + w_term
            x = torch.fft.ifft2(rhs / denom.clamp_min(self.eps)).real

            dx = diff_x(x)
            dy = diff_y(x)
            hx = torch.fft.ifft2(hf * torch.fft.fft2(x)).real

            alpha1 = alpha1 + mu1_i * (hx - v)
            alpha2_x = alpha2_x + mu2_i * (dx - u_x)
            alpha2_y = alpha2_y + mu2_i * (dy - u_y)
            alpha3 = alpha3 + mu3_i * (x - w)

        reconstruction_full = crop_center(w, sensor_height, sensor_width)
        reconstruction = reconstruction_full[
            ...,
            self.roi_top : self.roi_top + self.roi_height,
            self.roi_left : self.roi_left + self.roi_width,
        ]

        outputs = {"reconstruction": reconstruction}
        if self.return_full:
            outputs["reconstruction_full"] = reconstruction_full
        return outputs
