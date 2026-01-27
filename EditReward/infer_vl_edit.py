import os
import sys

# Add project root to Python path (optional, for local development)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from EditReward.inference_vl_edit import EditRewardVLInferencer


# ------------------------------------------------------------------------------
# Example script for evaluating edited images with EditReward (Unified VL)
# ------------------------------------------------------------------------------

CHECKPOINT_PATH = "your/local/path/to/checkpoint"
# Use either:
# - "config/EditReward-Qwen2.5-7B-VL.yaml"
# - "config/EditReward-Qwen3-VL.yaml"
CONFIG_PATH = "config/EditReward-Qwen3-VL.yaml"

inferencer = EditRewardVLInferencer(
    config_path=CONFIG_PATH,
    checkpoint_path=CHECKPOINT_PATH,
    device="cuda",
    reward_dim="overall_detail",
    rm_head_type="ranknet_multi_head",
)

image_src = [
    "your/local/path/to/source_image_1.jpg",
    "your/local/path/to/source_image_2.jpg",
]
image_paths = [
    "your/local/path/to/edited_image_1.jpg",
    "your/local/path/to/edited_image_2.jpg",
]
prompts = [
    "your first editing instruction",
    "your second editing instruction",
]

if __name__ == "__main__":
    with torch.no_grad():
        rewards = inferencer.reward(prompts=prompts, image_src=image_src, image_paths=image_paths)
    scores = [float(r[0]) if hasattr(r, "__len__") else float(r) for r in rewards]
    print(f"[Unified VL Pairwise Inference] Image scores: {scores}")


