from pathlib import Path
import torch


root_path = Path(__file__).resolve().parent.parent
device = "cpu"
model_config_dir = root_path / "src" / "configs" / "model"
models = [
    ("ADMM-100", "admm_100", None),
    ("Unrolled ADMM-20", "unrolled_admm_20", root_path / "saved" / "unrolled_admm_20" / "unrolled_admm_20_best.pth"),
    ("LeADMM-5 Pre", "leadmm5_pre", root_path / "saved" / "leadmm5_pre" / "leadmm5_pre_best.pth"),
    ("LeADMM-5 Post", "leadmm5_post", root_path / "saved" / "leadmm5_post" / "leadmm5_post_best.pth"),
    ("LeADMM-5 Pre+Post", "leadmm5_pre_post", root_path / "saved" / "leadmm5_pre_post" / "leadmm5_pre_post_best.pth")
]
