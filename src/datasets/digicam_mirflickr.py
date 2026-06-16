import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from src.datasets.base_dataset import BaseDataset
from src.lensless_helpers.preprocessor import get_dataset_object


class DigiCamMirflickrDataset(BaseDataset):
    def __init__(
        self,
        split,
        data_dir=None,
        limit=None,
        shuffle_index=False,
        instance_transforms=None
    ):
        self.repo_id = "bezzam/DigiCam-Mirflickr-MultiMask-10K"
        self.data_dir = data_dir
        self.dataset = load_dataset(
            "bezzam/DigiCam-Mirflickr-MultiMask-10K", 
            split=split, 
            cache_dir=data_dir
        )
        self.mask_paths = {}
        index = self._create_index(len(self.dataset))
        super().__init__(
            index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms
        )

    def load_object(self, data_dict):
        sample = self.dataset[data_dict["hf_index"]]
        mask_label = int(sample["mask_label"])
        mask_vals = np.load(self.get_mask_path(mask_label))
        lensed, lensless, psf = get_dataset_object(
            sample["lensed"],
            sample["lensless"],
            mask_vals
        )
        return self.build_item(data_dict["id"], lensless, lensed, psf, has_lensed=True)

    @staticmethod
    def _create_index(dataset_length):
        return [
            {
                "id": f"{i:06d}",
                "hf_index": i,
                "has_lensed": True
            }
            for i in range(dataset_length)
        ]

    def get_mask_path(self, mask_label):
        if mask_label not in self.mask_paths:
            self.mask_paths[mask_label] = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=f"masks/mask_{mask_label}.npy",
                cache_dir=self.data_dir
            )
        return self.mask_paths[mask_label]
