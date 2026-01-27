## Training Guideline

### 🤖 Model Support

- [x] **Qwen2.5-VL Series** 
- [x] **MiMo-VL Series**
- [x] **Qwen3-VL Series**

### Download EditReward-Data
<!-- ```
HPDv3 is comming soon! Stay tuned!
``` -->
```bash
huggingface-cli download --repo-type dataset TIGER-Lab/EditReward-Data --local-dir /your-local-dataset-path
```

### Prepare Pairwise Training Data Format

**Important Note: For simplicity, path1's image is always the prefered one**

#### Train set (`train.json`)
You should prepare a training dataset json follow below format:
```bash
[
    # samples from EditReward-Data
    {
    "prompt": "Make the sandwich a hot dog.",
    "choice_dist": null,
    "confidence": null,
    "path_src": "eff593b638c8c4b0aa385e8d2b5dcfc4/source.png",
    "path1": "eff593b638c8c4b0aa385e8d2b5dcfc4/ovis_u1_data_gen.png",               # Preferred sample
    "path2": "eff593b638c8c4b0aa385e8d2b5dcfc4/qwen_edit_seed2025_data_gen.png",    # Unpreferred sample
    "model1": "ovis_u1",                                                            # Model used to generate the preferred sample (path1).
    "model2": "qwen_edit_seed2025",                                                 # Model used to generate the non-preferred sample (path2).
    "model1_dim1_score": 4,
    "model1_dim2_score": 2,
    "model1_overall_score": 6,
    "model2_dim1_score": 4,
    "model2_dim2_score": 2,
    "model2_overall_score": 6,
    "tie_case": true
  },
  {
    "prompt": "Complete the image as you can",
    "choice_dist": null,
    "confidence": null,
    "path_src": "fffa04135540ff38e5064a8c6954dae40466522f5a0d6b06426fb1673d120b62/source.png",
    "path1": "fffa04135540ff38e5064a8c6954dae40466522f5a0d6b06426fb1673d120b62/step1x_v2n_seed42_data_gen.png",
    "path2": "fffa04135540ff38e5064a8c6954dae40466522f5a0d6b06426fb1673d120b62/qwen_edit_seed2025_data_gen.png",
    "model1": "step1x_v2n_seed42",
    "model2": "qwen_edit_seed2025",
    "model1_dim1_score": 1,
    "model1_dim2_score": 3,
    "model1_overall_score": 4,
    "model2_dim1_score": 1,
    "model2_dim2_score": 2,
    "model2_overall_score": 3,
    "tie_case": false
  },
  {
    "prompt": "Make the donut an apple.",
    "choice_dist": null,
    "confidence": null,
    "path_src": "139724552af3a46f8a27e9825a7fcea0/source.png",
    "path1": "139724552af3a46f8a27e9825a7fcea0/ovis_u1_seed2025_data_gen.png",
    "path2": "139724552af3a46f8a27e9825a7fcea0/flux_kontext_seed2025_data_gen.png",
    "model1": "ovis_u1_seed2025",
    "model2": "flux_kontext_seed2025",
    "model1_dim1_score": 4,
    "model1_dim2_score": 3,
    "model1_overall_score": 7,
    "model2_dim1_score": 3,
    "model2_dim2_score": 3,
    "model2_overall_score": 6,
    "tie_case": false
  },
]
```

#### Test Set (`test.json`) (**Optional**)
We prepare the GenAI-Bench valid dataset json in `data/dataset/valid_dataset_1.json`.
```bash
[   
    {
        "prompt": "let a herd of sheep block the taxi",
        "choice_dist": null,
        "confidence": null,
        "path_src": "GenAI-Bench/image_edition/images/50455b5ab8ef46578eec963b3fbe59f9_src.jpg",
        "path1": "GenAI-Bench/image_edition/images/50455b5ab8ef46578eec963b3fbe59f9_out.jpg",       # Preferred sample
        "path2": "GenAI-Bench/image_edition/images/f5b7def631bd4e7c8736368688baced7_out.jpg",       # Unpreferred sample
        "model1": "MagicBrush",                                                                     # Model used to generate the preferred sample (path1).
        "model2": "Pix2PixZero",                                                                    # Model used to generate the non-preferred sample (path2).
        "model1_dim1_score": 4,
        "model2_dim1_score": 2,
        "model1_overall_score": 0.9016961262134842,
        "model2_overall_score": 0.08793550441813658
    }
]
```

Then modify the path in `config/*.yaml`:
```bash
train_json_list:
  - /your/local/path/train.json
test_json_list:
  - ["Valid Set 1", ["/your/local/path/valid_set1.json"], "Valid Set 2", ["/your/local/path/valid_set2.json"]]
```


## 🏋️ Training

### 🚀 Training Command

```bash
git clone https://github.com/TIGER-AI-Lab/EditReward.git
cd EditReward

conda create -n edit_reward python=3.10 -y
conda activate edit_reward
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install datasets pillow openai -U megfile sentencepiece deepspeed fire omegaconf matplotlib peft trl==0.8.6 tensorboard scipy transformers==4.57.0 accelerate
# Recommend: Install flash-attn
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# (Recommended) Unified entry for Qwen2.5-VL / Qwen3-VL:
# It will automatically select model/collator based on `model_name_or_path` in the config.

# Train with Qwen2.5-7B-VL model (unified)
deepspeed EditReward/train_qwen_vl_edit.py --config EditReward/config/EditReward-Qwen2.5-7B-VL.yaml

# Train with Qwen3-VL model (unified)
deepspeed EditReward/train_qwen_vl_edit.py --config EditReward/config/EditReward-Qwen3-VL.yaml

# Train with Qwen2.5-7B-VL model
deepspeed EditReward/train_qwen2_5_edit.py --config EditReward/config/EditReward-Qwen2.5-7B-VL.yaml

# Train with MiMo-VL-7B-SFT-2508 model
deepspeed EditReward/train_qwen2_5_edit.py --config EditReward/config/EditReward-MiMo-VL-7B-SFT-2508.yaml

# (Optional) Train with Qwen3-VL model (Qwen3-only entry)
deepspeed EditReward/train_qwen3_vl_edit.py --config EditReward/config/EditReward-Qwen3-VL.yaml
```