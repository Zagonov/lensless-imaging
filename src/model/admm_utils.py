import torch
import torch.nn.functional as F
from scipy.fft import next_fast_len


def psf_to_fourier(psf):
    psf = torch.fft.ifftshift(psf, dim=(-2, -1))
    return torch.fft.fft2(psf)


def get_fft_size(size, padded_scale=2):
    return next_fast_len(padded_scale * size)


def center_pad(x, target_height, target_width):
    pad_height = target_height - x.shape[-2]
    pad_width = target_width - x.shape[-1]

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


def crop_center(x, target_height, target_width):
    top = (x.shape[-2] - target_height) // 2
    left = (x.shape[-1] - target_width) // 2
    return x[..., top : top + target_height, left : left + target_width]


def pad_to_shape(x, target_height, target_width):
    return center_pad(x, target_height, target_width)


def make_diff_otfs(height, width, device, dtype):
    kernel_x = torch.zeros(height, width, device=device, dtype=dtype)
    kernel_y = torch.zeros(height, width, device=device, dtype=dtype)

    kernel_x[0, 0] = -1
    kernel_x[0, 1] = 1

    kernel_y[0, 0] = -1
    kernel_y[1, 0] = 1

    dxf = torch.fft.fft2(kernel_x)
    dyf = torch.fft.fft2(kernel_y)

    return dxf.view(1, 1, height, width), dyf.view(1, 1, height, width)


def diff_x(x):
    return torch.roll(x, shifts=-1, dims=-1) - x


def diff_y(x):
    return torch.roll(x, shifts=-1, dims=-2) - x


def shrink(x, threshold):
    return torch.sign(x) * torch.clamp(x.abs() - threshold, min=0)
