from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.datasets.base_dataset import BaseDataset
from src.lensless_helpers.preprocessor import (
    convert_image_to_float,
    force_rgb,
    get_dataset_object,
)
from src.lensless_helpers.psf import simulate_psf_from_mask


class CustomDirDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        self.data_dir = Path(data_dir)
        index = self._create_index(self.data_dir)
        super().__init__(
            index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def load_object(self, data_dict):
        lensless = Image.open(data_dict["lensless_path"]).convert("RGB")
        mask_vals = np.load(data_dict["mask_path"])

        if data_dict["has_lensed"]:
            lensed = Image.open(data_dict["lensed_path"]).convert("RGB")
            lensed, lensless, psf = get_dataset_object(lensed, lensless, mask_vals)
            return self.build_item(
                data_dict["id"],
                lensless,
                lensed,
                psf,
                has_lensed=True,
            )

        lensless = convert_image_to_float(force_rgb(np.array(lensless)))
        lensless = torch.rot90(torch.from_numpy(lensless), dims=(-3, -2), k=2)
        lensed = torch.zeros_like(lensless)
        psf = simulate_psf_from_mask(mask_vals)
        return self.build_item(data_dict["id"], lensless, lensed, psf, has_lensed=False)

    @staticmethod
    def _create_index(data_dir):
        lensless_dir = data_dir / "lensless"
        mask_dir = data_dir / "masks"
        lensed_dir = data_dir / "lensed"

        index = []
        for lensless_path in sorted(lensless_dir.glob("*.png")):
            sample_id = lensless_path.stem
            lensed_path = lensed_dir / f"{sample_id}.png"
            index.append(
                {
                    "id": sample_id,
                    "lensless_path": str(lensless_path),
                    "mask_path": str(mask_dir / f"{sample_id}.npy"),
                    "lensed_path": str(lensed_path),
                    "has_lensed": lensed_path.exists(),
                }
            )
        return index
