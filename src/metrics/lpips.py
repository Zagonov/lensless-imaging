from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.metrics.base_metric import BaseMetric


class LPIPSMetric(BaseMetric):
    def __init__(self, name="LPIPS"):
        super().__init__(name=name)
        self.metric = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg",
            normalize=True
        )

    def __call__(self, reconstruction, lensed_roi, **batch):
        reconstruction = reconstruction.detach().cpu().clamp(0, 1)
        lensed_roi = lensed_roi.detach().cpu().clamp(0, 1)
        value = self.metric(reconstruction, lensed_roi)
        self.metric.reset()
        return value.item()
