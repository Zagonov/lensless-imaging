import random

import torch
from torch.utils.data import Dataset

from src.lensless_helpers.preprocessor import get_roi


class BaseDataset(Dataset):
    def __init__(
        self, index, limit=None, shuffle_index=False, instance_transforms=None
    ):
        self._assert_index_is_valid(index)
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index = index
        self.instance_transforms = instance_transforms or {}

    def __getitem__(self, ind):
        instance_data = self.load_object(self._index[ind])
        return self.preprocess_data(instance_data)

    def __len__(self):
        return len(self._index)

    def load_object(self, data_dict):
        raise NotImplementedError

    def preprocess_data(self, instance_data):
        for transform_name, transform in self.instance_transforms.items():
            if transform_name in instance_data:
                instance_data[transform_name] = transform(instance_data[transform_name])
        return instance_data

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "id" in entry
            assert "has_lensed" in entry

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)
        if limit is not None:
            index = index[:limit]
        return index

    @staticmethod
    def to_chw(image):
        if not torch.is_tensor(image):
            image = torch.from_numpy(image)
        return image.float().permute(2, 0, 1).contiguous()

    def build_item(self, sample_id, lensless, lensed, psf, has_lensed):
        if psf.ndim == 4:
            psf = psf[0]

        return {
            "id": str(sample_id),
            "lensless": self.to_chw(lensless),
            "lensed": self.to_chw(lensed),
            "lensed_roi": self.to_chw(get_roi(lensed)),
            "psf": self.to_chw(psf),
            "has_lensed": torch.tensor(has_lensed),
        }
