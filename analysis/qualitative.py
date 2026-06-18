import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from analysis.common import device, model_config_dir, models
from src.datasets import DigiCamMirflickrDataset


def collect_examples(num_examples=3):
    dataset = DigiCamMirflickrDataset(split="test", data_dir=".cache")
    indices = np.linspace(0, len(dataset) - 1, num_examples, dtype=int).tolist()
    loaded_models = []

    for model_name, config_name, checkpoint_path in models:
        config = OmegaConf.load(model_config_dir / f"{config_name}.yaml")
        model = instantiate(config).to(device)
        if checkpoint_path is not None:
            checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
        model.eval()
        loaded_models.append((model_name, model))

    examples = []
    for idx in indices:
        item = dataset[idx]
        lensless = item["lensless"].unsqueeze(0).to(device)
        psf = item["psf"].unsqueeze(0).to(device)

        reconstructions = {}
        with torch.no_grad():
            for model_name, model in loaded_models:
                output = model(lensless=lensless, psf=psf)
                reconstruction = output["reconstruction"][0]
                reconstruction = reconstruction.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                reconstructions[model_name] = reconstruction

        original = item["lensed_roi"].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        lensless_image = item["lensless"].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

        examples.append({
            "id": item["id"],
            "original": original,
            "lensless": lensless_image,
            "reconstructions": reconstructions
        })

    return examples


def show_qualitative_examples(num_examples=3):
    examples = collect_examples(num_examples=num_examples)
    model_names = list(examples[0]["reconstructions"].keys())

    n_rows = len(examples)
    n_cols = 2 + len(model_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)

    for row, example in enumerate(examples):
        axes[row, 0].imshow(example["original"])
        axes[row, 0].set_title(f"{example['id']} original")
        axes[row, 1].imshow(example["lensless"])
        axes[row, 1].set_title("lensless")

        for col, model_name in enumerate(model_names, start=2):
            axes[row, col].imshow(example["reconstructions"][model_name])
            axes[row, col].set_title(model_name)

        for axis in axes[row]:
            axis.axis("off")

    plt.tight_layout()


def show_error_maps(num_examples=3):
    examples = collect_examples(num_examples=num_examples)
    model_names = list(examples[0]["reconstructions"].keys())

    fig, axes = plt.subplots(len(examples), len(model_names), figsize=(4 * len(model_names), 4 * len(examples)))
    axes = np.atleast_2d(axes)

    for row, example in enumerate(examples):
        original = example["original"]
        for col, model_name in enumerate(model_names):
            reconstruction = example["reconstructions"][model_name]
            error = np.abs(reconstruction - original).mean(axis=-1)
            axes[row, col].imshow(error, cmap="magma")
            axes[row, col].set_title(f"{example['id']} {model_name}")
            axes[row, col].axis("off")

    plt.tight_layout()
