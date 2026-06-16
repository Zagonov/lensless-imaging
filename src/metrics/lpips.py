from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.metrics.base_metric import BaseMetric


class LPIPSMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(name="LPIPS", *args, **kwargs)
        self.metric = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg",
            normalize=True
        )

    def __call__(self, reconstruction, lensed_roi, **batch):
        reconstruction = reconstruction.detach().cpu()
        lensed_roi = lensed_roi.detach().cpu()
        value = self.metric(reconstruction, lensed_roi)
        self.metric.reset()
        return value.item()
