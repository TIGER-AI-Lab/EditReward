<p align="center" width="100%">
<img src="./assets/logo.png"  width="50%">
</p>

<div align="center">

# EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing

 ${{\color{Red}\Huge{\textsf{  ICLR\ 2026\ \}}}}\$
 
[![Project Website](https://img.shields.io/badge/🌐-Project%20Website-deepgray)](https://tiger-ai-lab.github.io/EditReward/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.26346-b31b1b.svg)](https://arxiv.org/abs/2509.26346)
[![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/collections/TIGER-Lab/editreward-68ddf026ef9eb1510458abc6)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-green)](https://huggingface.co/datasets/TIGER-Lab/EditReward-Data)
[![Benchmark](https://img.shields.io/badge/📊-Benchmark-yello)](https://huggingface.co/datasets/TIGER-Lab/EditReward-Bench)
</div>


## 📖 Introduction

This is the official implementation for the paper: [EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing](https://arxiv.org/abs/2508.03789).
In this paper, we introduce **EditReward**, a human-aligned reward model powered by a high-quality dataset for instruction-guided image editing. We first construct **EditReward-Data**, a large-scale, high-fidelity preference dataset for instruction-guided image editing. It comprises over 200K manually annotated preference pairs, covering a diverse range of edits produced by seven state-of-the-art models across twelve distinct sources. Every preference annotation in **EditReward-Data** was curated by trained annotators following a rigorous and standardized protocol, ensuring high alignment with considered human judgment and minimizing label noise. Using this dataset, we train the reward model **EditReward** to score instruction-guided image edits. To rigorously assess **EditReward** and future models, we also introduce **EditReward-Bench** a new benchmark built upon our high-quality annotations, which includes more difficult multi-way preference prediction.

<p align="center">
  <img src="assets/pipeline.png" alt="Teaser" width="900"/>
</p>


## 📰 News
- **[2026-01-27]** 🔥 Add training & inference support for **Qwen3-VL Series**!
- **[2026-01-26]** 🔥 Our paper has been accepted by **ICLR 2026**!
- **[2025-10-29]** 🔥 Release the training guideline of EditReward, see [Training Insctruction](EditReward/TRAIN_README.md)!
- **[2025-10-14]** 🔥 Release the evaluation code and guideline of EditReward-Bench, see [Evaluate Insctruction](EditReward/evaluate/README.md)!
- **[2025-10-10]** 🔥 Release our evaluation benchmark EditReward-Bench, Welcome to use!
- **[2025-10-08]** 🔥 Release our training dataset EditReward-Data, Welcome to use!
- **[2025-10-03]** 🔥 Release inference code and pretrained model.
- **[2025-10-01]** 🎉 We initialize the official repo of EditReward.

<!-- TODO List -->
## 🚧 TODO List
- [x] Release inference code and pretrained model
- [x] Release evaluation benchmark
- [x] Release training code
- [x] Release training dataset
<!-- - [ ] Release better model -->

## 📄 Table of Contents
- [🛠️ Installation](#%EF%B8%8F-installation-)
- [👨‍🏫 Get Started](#-get-started-)
- [🏋️ Training](#🏋️-training)
- [📊 Benchmark](#📊-benchmark)
- [🖊️ Citation](#%EF%B8%8F-citation-)
- [🤝 Acknowledgement](#-acknowledgement-)
- [🎫 License](#-license-)

---

## 🚀 Quick Start

EditReward is a VLM-based reward model trained on EditReward-Data that demonstrates superior alignment with human preferences.

### 💻 Installation

<!-- # Method 1: Pypi download and install for inference.
pip install hpsv3 -->

```bash

git clone https://github.com/TIGER-AI-Lab/EditReward.git
cd EditReward

conda create -n edit_reward python=3.10 -y
conda activate edit_reward
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install datasets pillow openai -U megfile sentencepiece deepspeed fire omegaconf matplotlib peft trl==0.8.6 tensorboard scipy transformers==4.56.1 accelerate
# Recommend: Install flash-attn
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

```


### 🚀 Usage

#### Basic Command
```python
import os
import sys
# Add project root to Python path (optional, for local development)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from EditReward import EditRewardInferencer
from EditReward.inference_vl_edit import EditRewardVLInferencer

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

# (Optional) Unified inferencer for Qwen2.5-VL / Qwen3-VL:
# Just switch CONFIG_PATH to either:
# - "config/EditReward-Qwen2.5-7B-VL.yaml"
# - "config/EditReward-Qwen3-VL.yaml"
# inferencer = EditRewardVLInferencer(
#     config_path=CONFIG_PATH,
#     checkpoint_path=CHECKPOINT_PATH,
#     device="cuda",
#     reward_dim="overall_detail",
#     rm_head_type="ranknet_multi_head",
# )

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
```

---


## 📁 Dataset

### EditReward-Data
<p align="left">
  <img src="assets/dataset_stat.png" alt="dataset" width="900"/>
</p>
<!-- <details close> -->

### Download EditReward
<!-- ```
HPDv3 is comming soon! Stay tuned!
``` -->
```bash
huggingface-cli download --repo-type dataset TIGER-Lab/EditReward-Data --local-dir /your-local-dataset-path
```

## 🏋️ Training

### 🤖 Model Support

- [x] **Qwen2.5-VL Series** 
- [x] **MiMo-VL Series**
- [x] **Qwen3-VL Series**

### 🚀 Training Command

To train **EditReward** model, follow the detail instruction in [Training Insctruction](EditReward/TRAIN_README.md)

#### Unified training entry (Qwen2.5-VL / Qwen3-VL)

We provide a unified training entry that automatically selects the correct model/collator based on `model_name_or_path`:

```bash
# Qwen2.5-VL
python EditReward/EditReward/train_qwen_vl_edit.py --config EditReward/EditReward/config/EditReward-Qwen2.5-7B-VL.yaml

# Qwen3-VL
python EditReward/EditReward/train_qwen_vl_edit.py --config EditReward/EditReward/config/EditReward-Qwen3-VL.yaml
```

---

## 📊 Benchmark
To evaluate **EditReward preference accuracy**, follow the detail instruction in [Evaluate Insctruction](EditReward/evaluate/README.md)

<details open>

<summary> Experimental Results: Alignment with Humans </summary>

| Method | GenAI-Bench | AURORA-Bench | ImagenHub | EditReward-Bench (Overall) |
| :--- | :--- | :--- | :--- | :--- |
| Random | 25.90 | 33.43 | -- | 13.84 |
| Human-to-Human | -- | -- | 41.84 | -- |
| ***Proprietary Models*** | | | | |
| GPT-4o | 53.54 | 50.81 | 38.21 | 28.31 |
| GPT-5 | 59.61 | 47.27 | <u>40.85</u> | 37.81 |
| Gemini-2.0-Flash | 53.32 | 44.31 | 23.69 | 33.47 |
| Gemini-2.5-Flash | 57.01 | 47.63 | **41.62** | <u>38.02</u> |
| ***Open-Source VLMs*** | | | | |
| Qwen2.5-VL-3B-Inst | 42.76 | 30.69 | -2.54 | 26.86 |
| Qwen2.5-VL-7B-Inst | 40.48 | 38.62 | 18.59 | 29.75 |
| Qwen2.5-VL-32B-Inst | 39.28 | 37.06 | 26.87 | 28.72 |
| MiMo-VL-7B-SFT-2508 | 57.89 | 30.43 | 22.14 | 31.19 |
| ADIEE | 59.96 | 55.56 | 34.50 | -- |
| ***Reward Models (Ours)*** | | | | |
| EditReward (on Qwen2.5-VL-7B) | <u>63.97</u> | <u>59.50</u> | 36.18 | 36.78 |
| EditReward (on MiMo-VL-7B) | **65.72** | **63.62** | 35.20 | **38.42** |
</details>

---

<details open>

<summary> EditReward-Bench Results </summary>

| Method | EditReward-Bench (K=2) | EditReward-Bench (K=3) | EditReward-Bench (K=4) | EditReward-Bench (Overall) |
| :--- | :--- | :--- | :--- | :--- |
| Random | 25.81 | 11.33 | 1.35 | 13.84 |
| Human-to-Human | -- | -- | -- | -- |
| ***Proprietary Models*** | | | | |
| GPT-4o | 45.69 | 27.33 | 7.31 | 28.31 |
| GPT-5 | <u>57.53</u> | 38.51 | <u>12.84</u> | 37.81 |
| Gemini-2.0-Flash | 52.43 | 33.33 | **13.51** | 33.47 |
| Gemini-2.5-Flash | **58.61** | <u>39.86</u> | 12.16 | <u>38.02</u> |
| ***Open-Source VLMs*** | | | | |
| Qwen2.5-VL-3B-Inst | 51.07 | 20.27 | 2.71 | 26.86 |
| Qwen2.5-VL-7B-Inst | 52.69 | 24.67 | 3.38 | 29.75 |
| Qwen2.5-VL-32B-Inst | 50.54 | 25.27 | 4.05 | 28.72 |
| MiMo-VL-7B-SFT-2508 | 49.46 | 30.41 | 9.46 | 31.19 |
| ADIEE | -- | -- | -- | -- |
| ***Reward Models (Ours)*** | | | | |
| EditReward (on Qwen2.5-VL-7B) | 56.99 | 36.00 | 10.81 | 36.78 |
| EditReward (on MiMo-VL-7B) | 56.45 | **42.67** | 11.49 | **38.42** |
</details>

---


## 📚 Citation

Please kindly cite our paper if you use our code, data, models or results:

```bibtex
@article{wu2025editreward,
  title={EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing},
  author={Wu, Keming and Jiang, Sicong and Ku, Max and Nie, Ping and Liu, Minghao and Chen, Wenhu},
  journal={arXiv preprint arXiv:2509.26346},
  year={2025}
}
```


---

## 🙏 Acknowledgements

We would like to thank the [HPSv3](https://github.com/MizzenAI/HPSv3), [VideoAlign](https://github.com/KwaiVGI/VideoAlign) and [GenAI-Bench](https://github.com/TIGER-AI-Lab/GenAI-Bench) codebase for providing valuable references.

---
## ⭐ Star History [🔝](#-table-of-contents)

[![Star History Chart](https://api.star-history.com/svg?repos=TIGER-AI-Lab/EditReward&type=Date)](https://star-history.com/#TIGER-AI-Lab/EditReward&Date)
## 💬 Support

For questions and support:
- **Issues**: [GitHub Issues](https://github.com/TIGER-AI-Lab/EditReward/issues)
- **Email**: wukeming0608@gmail.com & wenhuchen@uwaterloo.ca
