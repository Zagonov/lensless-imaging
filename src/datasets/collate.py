import torch


def collate_fn(items):
    return {
        "id": [item["id"] for item in items],
        "lensless": torch.stack([item["lensless"] for item in items]),
        "lensed": torch.stack([item["lensed"] for item in items]),
        "lensed_roi": torch.stack([item["lensed_roi"] for item in items]),
        "psf": torch.stack([item["psf"] for item in items]),
        "has_lensed": torch.stack([item["has_lensed"] for item in items]),
    }


lensless_collate_fn = collate_fn
