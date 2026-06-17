from torchmetrics.regression import MeanSquaredError

from src.metrics.base_metric import BaseMetric


class MSEMetric(BaseMetric):
    def __init__(self, name="MSE"):
        super().__init__(name=name)
        self.metric = MeanSquaredError()

    def __call__(self, reconstruction, lensed_roi, **batch):
        reconstruction = reconstruction.detach().cpu().contiguous()
        lensed_roi = lensed_roi.detach().cpu().contiguous()
        value = self.metric(reconstruction, lensed_roi)
        self.metric.reset()
        return value.item()
