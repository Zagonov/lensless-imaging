from torch import nn
from src.metrics.base_metric import BaseMetric


class MSEMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(name="MSE", *args, **kwargs)
        self.metric = nn.MSELoss()

    def __call__(self, reconstruction, lensed_roi, **batch):
        reconstruction = reconstruction.detach().cpu()
        lensed_roi = lensed_roi.detach().cpu()
        value = self.metric(reconstruction, lensed_roi)
        self.metric.reset()
        return value.item()
