from torch import nn


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, reconstruction, lensed_roi, **batch):
        loss = self.loss(reconstruction, lensed_roi)
        return {"loss": loss}
