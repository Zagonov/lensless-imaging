from torchmetrics.image import PeakSignalNoiseRatio

from src.metrics.base_metric import BaseMetric


class PSNRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(name="PSNR", *args, **kwargs)
        self.metric = PeakSignalNoiseRatio(data_range=1)

    def __call__(self, reconstruction, lensed_roi, **batch):
        reconstruction = reconstruction.detach().cpu()
        lensed_roi = lensed_roi.detach().cpu()
        value = self.metric(reconstruction, lensed_roi)
        self.metric.reset()
        return value.item()
