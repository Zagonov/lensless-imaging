import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
from hydra.utils import instantiate
from IPython.display import display
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from analysis.common import device, model_config_dir, models
from src.datasets import DigiCamMirflickrDataset
from src.metrics import LPIPSMetric, MSEMetric, PSNRMetric, SSIMMetric


def get_per_image_metrics(num_samples=None):
    dataset = DigiCamMirflickrDataset(split="test", data_dir=".cache", limit=num_samples)
    rows = []

    for model_name, config_name, checkpoint_path in models:
        config = OmegaConf.load(model_config_dir / f"{config_name}.yaml")
        model = instantiate(config).to(device)
        if checkpoint_path is not None:
            checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
        model.eval()

        mse = MSEMetric()
        psnr = PSNRMetric()
        ssim = SSIMMetric()
        lpips = LPIPSMetric()

        for item in tqdm(dataset, desc=model_name, total=len(dataset), leave=False):
            lensless = item["lensless"].unsqueeze(0).to(device)
            psf = item["psf"].unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(lensless=lensless, psf=psf)

            batch = {
                "reconstruction": output["reconstruction"].detach().cpu(),
                "lensed_roi": item["lensed_roi"].unsqueeze(0)
            }

            rows.append({
                "id": item["id"],
                "model": model_name,
                "MSE": mse(**batch),
                "PSNR": psnr(**batch),
                "SSIM": ssim(**batch),
                "LPIPS": lpips(**batch)
            })

    return pd.DataFrame(rows)


def show_metric_table(num_samples=None):
    rows = get_per_image_metrics(num_samples=num_samples)
    summary = rows.groupby("model")[["MSE", "PSNR", "SSIM", "LPIPS"]].mean().sort_values("PSNR", ascending=False)
    display(summary)


def show_metric_distributions(num_samples=None):
    rows = get_per_image_metrics(num_samples=num_samples)
    grouped = rows.groupby("model")
    labels = list(grouped.groups.keys())
    psnr_data = [group["PSNR"].values for _, group in grouped]
    lpips_data = [group["LPIPS"].values for _, group in grouped]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].boxplot(psnr_data, tick_labels=labels, vert=True)
    axes[0].set_title("PSNR distribution")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].boxplot(lpips_data, tick_labels=labels, vert=True)
    axes[1].set_title("LPIPS distribution")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()


def get_speed_table(num_samples, warmup):
    dataset = DigiCamMirflickrDataset(split="test", data_dir=".cache", limit=num_samples + warmup)
    items = [dataset[idx] for idx in range(len(dataset))]
    rows = []

    for model_name, config_name, checkpoint_path in models:
        config = OmegaConf.load(model_config_dir / f"{config_name}.yaml")
        model = instantiate(config).to(device)
        if checkpoint_path is not None:
            checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
        model.eval()

        for item in items[:warmup]:
            lensless = item["lensless"].unsqueeze(0).to(device)
            psf = item["psf"].unsqueeze(0).to(device)
            with torch.no_grad():
                model(lensless=lensless, psf=psf)

        start_time = time.perf_counter()
        for item in items[warmup:]:
            lensless = item["lensless"].unsqueeze(0).to(device)
            psf = item["psf"].unsqueeze(0).to(device)
            with torch.no_grad():
                model(lensless=lensless, psf=psf)
        elapsed = time.perf_counter() - start_time

        rows.append({
            "model": model_name,
            "sec_per_image": elapsed / num_samples,
            "images_per_sec": num_samples / elapsed
        })

    return pd.DataFrame(rows).sort_values("sec_per_image")


def show_speed_table(num_samples=20, warmup=2):
    speed_table = get_speed_table(num_samples, warmup)
    display(speed_table)
