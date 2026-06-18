import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from src.lensless_helpers.preprocessor import convert_image_to_float, force_rgb
from src.lensless_helpers.utils import resize
from src.metrics import LPIPSMetric, MSEMetric, PSNRMetric, SSIMMetric
from src.metrics.tracker import MetricTracker


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--predictions-dir", required=True)
    return parser.parse_args()


def load_image(path):
    image = np.array(Image.open(path).convert("RGB"))
    image = convert_image_to_float(force_rgb(image))
    return image


def to_tensor(image):
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()


def main():
    args = get_args()

    original_dir = Path(args.data_dir) / "lensed"
    predictions_dir = Path(args.predictions_dir)

    original_paths = {path.stem: path for path in original_dir.glob("*.png")}
    prediction_paths = {path.stem: path for path in predictions_dir.glob("*.png")}
    image_ids = sorted(set(original_paths) & set(prediction_paths))

    metrics = [MSEMetric(), PSNRMetric(), SSIMMetric(), LPIPSMetric()]
    tracker = MetricTracker(*[metric.name for metric in metrics], writer=None)

    for image_id in tqdm(image_ids, desc="metrics"):
        original = load_image(original_paths[image_id])
        prediction = load_image(prediction_paths[image_id])

        if original.shape != prediction.shape:
            original = resize(
                original, shape=prediction.shape, interpolation=cv2.INTER_NEAREST
            )

        batch = {
            "reconstruction": to_tensor(prediction),
            "lensed_roi": to_tensor(original),
        }

        for metric in metrics:
            tracker.update(metric.name, metric(**batch))

    print(f"num images: {len(image_ids)}")
    for name, value in tracker.result().items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
