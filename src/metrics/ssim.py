from torchmetrics.image import StructuralSimilarityIndexMeasure

from src.metrics.base_metric import BaseMetric


class SSIMMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(name="SSIM", *args, **kwargs)
        self.metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    def __call__(self, reconstruction, lensed_roi, **batch):
        reconstruction = reconstruction.detach().cpu()
        lensed_roi = lensed_roi.detach().cpu()
        value = self.metric(reconstruction, lensed_roi)
        self.metric.reset()
        return value.item()
