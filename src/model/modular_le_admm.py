from torch import nn

from src.model.drunet import DRUNet
from src.model.le_admm import LeADMM


class ModularLeADMM(nn.Module):
    def __init__(
        self,
        n_iter=5,
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
        use_preprocessor=False,
        use_postprocessor=False,
        pre_channels=(32, 64, 128, 256),
        post_channels=(32, 64, 128, 256),
        n_blocks=4,
    ):
        super().__init__()
        self.use_preprocessor = use_preprocessor
        self.use_postprocessor = use_postprocessor
        self.roi_top = roi_top
        self.roi_left = roi_left
        self.roi_height = roi_height
        self.roi_width = roi_width
        self.return_full = return_full

        if self.use_preprocessor:
            self.preprocessor = DRUNet(
                in_channels=3,
                out_channels=3,
                channels=pre_channels,
                n_blocks=n_blocks,
            )

        self.inversion = LeADMM(
            n_iter=n_iter,
            mu1_init=mu1_init,
            mu2_init=mu2_init,
            mu3_init=mu3_init,
            tau_init=tau_init,
            learnable=learnable,
            padded_scale=padded_scale,
            roi_top=roi_top,
            roi_left=roi_left,
            roi_height=roi_height,
            roi_width=roi_width,
            eps=eps,
            return_full=True,
        )

        if self.use_postprocessor:
            self.postprocessor = DRUNet(
                in_channels=3,
                out_channels=3,
                channels=post_channels,
                n_blocks=n_blocks,
            )

    def forward(self, lensless, psf, **batch):
        if self.use_preprocessor:
            lensless = self.preprocessor(lensless)

        outputs = self.inversion(lensless=lensless, psf=psf, **batch)
        reconstruction_full = outputs["reconstruction_full"]

        if self.use_postprocessor:
            reconstruction_full = self.postprocessor(reconstruction_full)

        reconstruction = reconstruction_full[
            ...,
            self.roi_top:self.roi_top + self.roi_height,
            self.roi_left:self.roi_left + self.roi_width,
        ]

        result = {"reconstruction": reconstruction}
        if self.return_full:
            result["reconstruction_full"] = reconstruction_full
        return result
