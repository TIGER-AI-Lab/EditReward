<p align="center" width="100%">
<img src="./assets/logo.png"  width="80%">
</p>

<div align="center">

# EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing

[![Project Website](https://img.shields.io/badge/🌐-Project%20Website-deepgray)](https://github.com/TIGER-AI-Lab/EditReward)
[![arXiv](https://img.shields.io/badge/arXiv-2508.03789-b31b1b.svg)](https://arxiv.org/abs/2508.03789)
<!-- [![ICCV 2025](https://img.shields.io/badge/ICCV-2025-blue.svg)](https://arxiv.org/abs/2508.03789) -->
[![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/MizzenAI/HPSv3)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-green)](https://huggingface.co/datasets/MizzenAI/HPDv3)
<!-- [![PyPI](https://img.shields.io/pypi/v/hpsv3)](https://pypi.org/project/hpsv3/) -->

<!-- **Yuhang Ma**<sup>1,3*</sup>&ensp; **Yunhao Shui**<sup>1,4*</sup>&ensp; **Xiaoshi Wu**<sup>2</sup>&ensp; **Keqiang Sun**<sup>1,2†</sup>&ensp; **Hongsheng Li**<sup>2,5,6†</sup> -->

<!-- <sup>1</sup>Mizzen AI&ensp;&ensp; <sup>2</sup>CUHK MMLab&ensp;&ensp; <sup>3</sup>King’s College London&ensp;&ensp; <sup>4</sup>Shanghai Jiaotong University&ensp;&ensp;  -->

<!-- <sup>5</sup>Shanghai AI Laboratory&ensp;&ensp; <sup>6</sup>CPII, InnoHK&ensp;&ensp;  -->

<!-- <sup>*</sup>Equal Contribution&ensp; <sup>†</sup>Equal Advising -->

</div>


## 📖 Introduction

This is the official implementation for the paper: [EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing](https://arxiv.org/abs/2508.03789).
In this paper, we introduce **EditReward**, a human-aligned reward model powered by a high-quality dataset for instruction-guided image editing. We first construct **EditReward-Data**, a large-scale, high-fidelity preference dataset for instruction-guided image editing. It comprises over 200K manually annotated preference pairs, covering a diverse range of edits produced by seven state-of-the-art models across twelve distinct sources. Every preference annotation in **EditReward-Data** was curated by trained annotators following a rigorous and standardized protocol, ensuring high alignment with considered human judgment and minimizing label noise. Using this dataset, we train the reward model **EditReward** to score instruction-guided image edits. To rigorously assess **EditReward** and future models, we also introduce **EditReward-Bench** a new benchmark built upon our high-quality annotations, which includes more difficult multi-way preference prediction.

<p align="center">
  <img src="assets/pipeline.png" alt="Teaser" width="900"/>
</p>


## 📰 News
- **[2025-10-01]** 🎉 We release EditReward: inference code, training code and [EditReward model weights](https://huggingface.co/MizzenAI/HPSv3). And [PyPI Package](https://pypi.org/project/hpsv3/).

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

HPSv3 is a state-of-the-art human preference score model for evaluating image quality and prompt alignment. It builds upon the Qwen2-VL architecture to provide accurate assessments of generated images.

### 💻 Installation

<!-- # Method 1: Pypi download and install for inference.
pip install hpsv3 -->

```bash
# Method 1: Pypi download and install for inference.
pip install hpsv3

# Method 2: Install locally for development or training.
git clone https://github.com/MizzenAI/HPSv3.git
cd HPSv3

conda env create -f environment.yaml
conda activate hpsv3
# Recommend: Install flash-attn
pip install flash-attn==2.7.4.post1

pip install -e .
```

### 🛠️ Basic Usage

#### Simple Inference Example

```python
from EditReward import EditRewardInferencer

# Initialize the model
inferencer = EditRewardInferencer(device='cuda')

# Evaluate images
image_paths = ["assets/example1.png", "assets/example2.png"]
prompts = [
  "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker",
  "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker"
]

# Get preference scores
rewards = inferencer.reward(prompts, image_paths=image_paths)
scores = [reward[0].item() for reward in rewards]  # Extract mu values
print(f"Image scores: {scores}")
```

---

## 🌐 Gradio Demo

Launch an interactive web interface to test HPSv3:

```bash
python gradio_demo/demo.py
```

The demo will be available at `http://localhost:7860` and provides:

<p align="left">
  <img src="assets/gradio.png" alt="Gradio Demo" width="500"/>
</p>



## 📁 Dataset

### Human Preference Dataset v3

Human Preference Dataset v3 (HPD v3) comprises 1.08M text-image pairs and 1.17M annotated pairwise data. To modeling the wide spectrum of human preference, we introduce newest state-of-the-art generative models and high quality real photographs while maintaining old models and lower quality real images.
<p align="left">
  <img src="assets/dataset_stat.pdf" alt="dataset" width="500"/>
</p>
<!-- <details close> -->

### Download EditReward
<!-- ```
HPDv3 is comming soon! Stay tuned!
``` -->
```bash
huggingface-cli download --repo-type dataset MizzenAI/HPDv3 --local-dir /your-local-dataset-path
```

### Pairwise Training Data Format

**Important Note: For simplicity, path1's image is always the prefered one**

#### All Annotated Pairs (`all.json`)

**Important Notes: In HPDv3, we simply put the preferred sample at the first place (path1)**

`all.json` contains **all** annotated pairs except for test.

```bash
[
    # samples from HPDv3 annotation pipeline 
    {
    "prompt": "Description of the visual content or the generation prompt.",
    "choice_dist": [12, 7],           # Distribution of votes from annotators (12 votes for image1, 7 votes for image2)
    "confidence": 0.9999907,         # Confidence score reflecting preference reliability, based on annotators' capabilities (independent of choice_dist)
    "path1": "images/uuid1.jpg",     # File path to the preferred image
    "path2": "images/uuid2.jpg",     # File path to the non-preferred image
    "model1": "flux",                # Model used to generate the preferred image (path1)
    "model2": "infinity"             # Model used to generate the non-preferred image (path2)
    },
    # samples from Midjourney
    {
    "prompt": "Description of the visual content or the generation prompt.",
    "choice_dist": null,             # No distribution of votes Information from Discord
    "confidence": null,              # No Confidence Information from Discord
    "path1": "images/uuid1.jpg",     # File path to the preferred image.
    "path2": "images/uuid2.jpg",     # File path to the non-preferred image.
    "model1": "midjourney",          # Comparsion between images generated from midjourney 
    "model2": "midjourney"           # Comparsion between images generated from midjourney 
    },
    # samples from Curated HPDv2
    {
    "prompt": "Description of the visual content or the generation prompt.",
    "choice_dist": null,              # No distribution of votes Information from the original HPDv2 traindataset
    "confidence": null,               # No Confidence Information from the original HPDv2 traindataset
    "path1": "images/uuid1.jpg",     # File path to the preferred image.
    "path2": "images/uuid2.jpg",     # File path to the non-preferred image.
    "model1": "hpdv2",          # No specific model name in the original HPDv2 traindataset, set to hpdv2 
    "model2": "hpdv2"           # No specific model name in the original HPDv2 traindataset, set to hpdv2 
    },
]
```

#### Train set (`train.json`)
We sample part of training data from `all.json` to build training dataset `train.json`. Moreover, to improve robustness, we integrate random sampled part of data from [Pick-a-pic](https://huggingface.co/datasets/pickapic-anonymous/pickapic_v1) and [ImageRewardDB](https://huggingface.co/datasets/zai-org/ImageRewardDB), which is `pickapic.json` and `imagereward.json`. For these two datasets, we only provide the pair infomation, and its corresponding image can be found in their official dataset repository.


#### Test Set (`test.json`)
```bash
[
    {
        "prompt": "Description of the visual content",
        "path1": "images/uuid1.jpg",     # Preferred sample
        "path2": "images/uuid2.jpg",     # Unpreferred sample
        "model1": "flux",                # Model used to generate the preferred sample (path1).
        "model2": "infinity",            # Model used to generate the non-preferred sample (path2).

    }
]
```

## 🏋️ Training

### 🚀 Training Command

```bash
# Use Method 2 to install locally
git clone https://github.com/MizzenAI/HPSv3.git
cd HPSv3

conda env create -f environment.yaml
conda activate hpsv3
# Recommend: Install flash-attn
pip install flash-attn==2.7.4.post1

pip install -e .

# Train with 7B model
deepspeed hpsv3/train.py --config hpsv3/config/HPSv3_7B.yaml
```

<details close>
<summary>Important Config Argument</summary>

| Configuration Section | Parameter | Value | Description |
|----------------------|-----------|-------|-------------|
| **Model Configuration** | `rm_head_type` | `"ranknet"` | Type of reward model head architecture |
| | `lora_enable` | `False` | Enable LoRA (Low-Rank Adaptation) for efficient fine-tuning. If `False`, language tower is fully trainable|
| | `vision_lora` | `False` | Apply LoRA specifically to vision components. If `False`, vision tower is fully trainable|
| | `model_name_or_path` | `"Qwen/Qwen2-VL-7B-Instruct"` | Path to the base model checkpoint |
| **Data Configuration** | `confidence_threshold` | `0.95` | Minimum confidence score for training data |
| | `train_json_list` | `[example_train.json]` | List of training data files |
| | `test_json_list` | `[validation_sets]` | List of validation datasets with names |
| | `output_dim` | `2` | Output dimension of the reward head for $\mu$ and $\sigma$|
| | `loss_type` | `"uncertainty"` | Loss function type for training |
</details>

---

## 📊 Benchmark
To evaluate **HPSv3 preference accuracy** or **human preference score of image generation model**, follow the detail instruction is in [Evaluate Insctruction](evaluate/README.md)

<details open>
<summary> Experimental Results: Alignment with Humans </summary>

Ah, my sincere apologies for the persistent misunderstanding. You want the table to include **only the "Overall" result from `\benchname`**, alongside the public benchmarks, without the `K=2, K=3, K=4` columns.

This makes the table much more compact.

Here is the corrected Markdown table, featuring only `GenAI-Bench`, `AURORA-Bench`, `ImagenHub`, and `\benchname Overall`.

-----

| Method | GenAI-Bench | AURORA-Bench | ImagenHub | EditReward-Bench (Overall) |
| :--- | :--- | :--- | :--- | :--- |
| Random | 25.90 | 33.43 | -- | 13.84 |
| Human-to-Human | -- | -- | 41.84 | -- |
| ***Proprietary Models*** | | | | |
| GPT-4o | 53.54 | 50.81 | 38.21 | 28.31 |
| GPT-5 | 59.61 | 47.27 | \<u\>40.85\</u\> | 37.81 |
| Gemini-2.0-Flash | 53.32 | 44.31 | 23.69 | 33.47 |
| Gemini-2.5-Flash | 57.01 | 47.63 | **41.62** | \<u\>38.02\</u\> |
| ***Open-Source VLMs*** | | | | |
| Qwen2.5-VL-3B-Inst | 42.76 | 30.69 | -2.54 | 26.86 |
| Qwen2.5-VL-7B-Inst | 40.48 | 38.62 | 18.59 | 29.75 |
| Qwen2.5-VL-32B-Inst | 39.28 | 37.06 | 26.87 | 28.72 |
| MiMo-VL-7B-SFT-2508 | 57.89 | 30.43 | 22.14 | 31.19 |
| ADIEE | 59.96 | 55.56 | 34.50 | -- |
| ***Reward Models (Ours)*** | | | | |
| EditReward (on Qwen2.5-VL-7B) | \<u\>63.97\</u\> | \<u\>59.50\</u\> | 36.18 | 36.78 |
| EditReward (on MiMo-VL-7B) | **65.72** | **63.62** | 35.20 | **38.42** |
</details>

<details open>
<summary> EditReward-Bench Results </summary>
-----

| Method | EditReward-Bench (K=2) | EditReward-Bench (K=3) | EditReward-Bench (K=4) | EditReward-Bench (Overall) |
| :--- | :--- | :--- | :--- | :--- |
| Random | 25.81 | 11.33 | 1.35 | 13.84 |
| Human-to-Human | -- | -- | -- | -- |
| ***Proprietary Models*** | | | | |
| GPT-4o | 45.69 | 27.33 | 7.31 | 28.31 |
| GPT-5 | \<u\>57.53\</u\> | 38.51 | \<u\>12.84\</u\> | 37.81 |
| Gemini-2.0-Flash | 52.43 | 33.33 | **13.51** | 33.47 |
| Gemini-2.5-Flash | **58.61** | \<u\>39.86\</u\> | 12.16 | \<u\>38.02\</u\> |
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


### 🚀 Usage

#### Basic Command



## 📚 Citation

Please kindly cite our paper if you use our code, data, models or results:

```bibtex

```


---

## 🙏 Acknowledgements

We would like to thank the [HPSv3](https://github.com/MizzenAI/HPSv3) and [VideoAlign](https://github.com/KwaiVGI/VideoAlign) codebase for providing valuable references.

---
## ⭐ Star History [🔝](#-table-of-contents)

[![Star History Chart](https://api.star-history.com/svg?repos=TIGER-AI-Lab/EditReward&type=Date)](https://star-history.com/#TIGER-AI-Lab/EditReward&Date)
## 💬 Support

For questions and support:
- **Issues**: [GitHub Issues](https://github.com/MizzenAI/HPSv3/issues)
- **Email**: yhshui@mizzen.ai & yhma@mizzen.ai