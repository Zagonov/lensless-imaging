from torch import nn

from src.loss.lpips_loss import LPIPSLoss
from src.loss.mse_loss import MSELoss


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        lambda_mse=1,
        lambda_lpips=1,
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_lpips = lambda_lpips
        self.mse_loss = MSELoss()
        self.lpips_loss = None
        if self.lambda_lpips != 0:
            self.lpips_loss = LPIPSLoss()

    def forward(self, reconstruction, lensed_roi, **batch):
        mse = self.mse_loss(reconstruction, lensed_roi)["loss"]
        lpips = mse.new_zeros(())
        if self.lpips_loss is not None:
            lpips = self.lpips_loss(reconstruction, lensed_roi)["loss"]
        loss = self.lambda_mse * mse + self.lambda_lpips * lpips

        return {
            "loss": loss,
            "mse_loss": mse,
            "lpips_loss": lpips,
        }
