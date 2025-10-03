import os
import sys
# Add project root to Python path (optional, for local development)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from EditReward import EditRewardInferencer

# ------------------------------------------------------------------------------
# Example script for evaluating edited images with EditReward
# ------------------------------------------------------------------------------

# Path to model checkpoint (update to your own local or HF path)
CHECKPOINT_PATH = "your/local/path/to/checkpoint"
CONFIG_PATH = "config/EditReward-MiMo-VL-7B-SFT-2508.yaml"

# Initialize reward model
inferencer = EditRewardInferencer(
    config_path=CONFIG_PATH,
    checkpoint_path=CHECKPOINT_PATH,
    device="cuda",        # or "cpu"
    reward_dim="overall_detail",    # choose reward dimension if applicable
    rm_head_type="ranknet_multi_head"
)

# Example input data -----------------------------------------------------------
# image_src = [
#     "../assets/examples/source_img_1.png",
#     "../assets/examples/source_img_1.png",
# ]

# image_paths = [
#     "../assets/examples/target_img_1.png",
#     "../assets/examples/target_img_2.png",
# ]
image_src = [
    "your/local/path/to/source_image_1.jpg",
    "your/local/path/to/source_image_2.jpg",
]

image_paths = [
    "your/local/path/to/edited_image_1.jpg",
    "your/local/path/to/edited_image_2.jpg",
]

# example instruction: "Add a green bowl on the branch"
# prompts = [
#     "Add a green bowl on the branch",
#     "Add a green bowl on the branch"
# ]
prompts = [
    "your first editing instruction",
    "your second editing instruction"
]

# ------------------------------------------------------------------------------
# Main evaluation modes
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    mode = "pairwise_inference"  # or "single_inference"

    if mode == "pairwise_inference":
        # ----------------------------------------------------------
        # Pairwise comparison: compares two edited images side-by-side
        # ----------------------------------------------------------
        with torch.no_grad():
          rewards = inferencer.reward(
              prompts=prompts,
              image_src=image_src,
              image_paths=image_paths
          )
        scores = [reward[0].item() for reward in rewards]
        print(f"[Pairwise Inference] Image scores: {scores}")

    elif mode == "single_inference":
        # ----------------------------------------------------------
        # Single image scoring: evaluates one edited image at a time
        # ----------------------------------------------------------
        with torch.no_grad():
          rewards = inferencer.reward(
              prompts=[prompts[0]],
              image_src=[image_src[0]],
              image_paths=[image_paths[0]]
          )
        print(f"[Single Inference] Image 1 score: {[reward[0].item() for reward in rewards]}")
        
        with torch.no_grad():
          rewards = inferencer.reward(
              prompts=[prompts[0]],
              image_src=[image_src[0]],
              image_paths=[image_paths[1]]
          )
        print(f"[Single Inference] Image 2 score: {[reward[0].item() for reward in rewards]}")
