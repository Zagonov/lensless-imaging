from torch import nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class LPIPSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg",
            normalize=True
        )

    def forward(self, reconstruction, lensed_roi, **batch):
        reconstruction = reconstruction.clamp(0, 1)
        lensed_roi = lensed_roi.clamp(0, 1)
        loss = self.loss(reconstruction, lensed_roi)
        return {"loss": loss}
