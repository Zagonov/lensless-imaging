import numpy as np
import torch
from huggingface_hub import hf_hub_download

from datasets import load_dataset
from src.datasets.base_dataset import BaseDataset
from src.lensless_helpers.preprocessor import (
    convert_image_to_float,
    force_rgb,
    get_cropped_lensed,
)
from src.lensless_helpers.psf import simulate_psf_from_mask


class DigiCamMirflickrDataset(BaseDataset):
    def __init__(
        self,
        split,
        data_dir=None,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        self.repo_id = "bezzam/DigiCam-Mirflickr-MultiMask-10K"
        self.data_dir = data_dir
        self.dataset = load_dataset(
            "bezzam/DigiCam-Mirflickr-MultiMask-10K", split=split, cache_dir=data_dir
        )
        self.mask_paths = {}
        self.psf_cache = {}
        index = self._create_index(len(self.dataset))
        super().__init__(
            index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def load_object(self, data_dict):
        sample = self.dataset[data_dict["hf_index"]]
        mask_label = int(sample["mask_label"])

        lensed = convert_image_to_float(force_rgb(np.array(sample["lensed"])))
        lensless = convert_image_to_float(force_rgb(np.array(sample["lensless"])))
        lensless = torch.rot90(torch.from_numpy(lensless), dims=(-3, -2), k=2)
        lensed = get_cropped_lensed(lensed, lensless)
        lensed = torch.from_numpy(lensed)

        if mask_label not in self.psf_cache:
            mask_vals = np.load(self.get_mask_path(mask_label))
            self.psf_cache[mask_label] = simulate_psf_from_mask(mask_vals)

        psf = self.psf_cache[mask_label]
        return self.build_item(data_dict["id"], lensless, lensed, psf, has_lensed=True)

    @staticmethod
    def _create_index(dataset_length):
        return [
            {"id": f"{i:06d}", "hf_index": i, "has_lensed": True}
            for i in range(dataset_length)
        ]

    def get_mask_path(self, mask_label):
        if mask_label not in self.mask_paths:
            self.mask_paths[mask_label] = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=f"masks/mask_{mask_label}.npy",
                cache_dir=self.data_dir,
            )
        return self.mask_paths[mask_label]
