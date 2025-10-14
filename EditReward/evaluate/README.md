# EditReward-Bench Evaluation Suite

This repository contains evaluation scripts for testing multimodal language models on the [EditReward-Bench](https://huggingface.co/datasets/TIGER-Lab/EditReward-Bench) dataset. The evaluation suite supports different model types and provides comprehensive metrics for image editing quality assessment.

## 🚀 Quick Start

### Prerequisites

```bash
pip install datasets huggingface_hub tqdm fire pillow
```

### Basic Usage

```bash
# Test with 2-pair data (186 samples)
python3 inference_2pair_hf.py --max_examples 5

# Full evaluation on 2-pair data
python3 inference_2pair_hf.py

# Test with 3-pair data (450 samples)
python3 inference_3pair_hf.py --max_examples 1

# Test with 4-pair data (888 samples)  
python3 inference_4pair_hf.py --max_examples 1
```

## 📊 Dataset Overview

The EditReward-Bench dataset contains **1,524 pairwise comparisons** organized as:

| Subset | Original Samples | Pairwise Comparisons | Description |
|--------|-----------------|---------------------|-------------|
| **2-pair** | 186 | 186 | 1 comparison per sample (A vs B) |
| **3-pair** | 150 | 450 | 3 comparisons per sample (A vs B, A vs C, B vs C) |
| **4-pair** | 148 | 888 | 6 comparisons per sample (A vs B, A vs C, A vs D, B vs C, B vs D, C vs D) |
| **Total** | **484** | **1,524** | All pairwise comparisons |

### Evaluation Metrics

- **Individual Accuracy**: Percentage of correct pairwise predictions
- **Group Accuracy**: For 3-pair and 4-pair, percentage of groups where ALL comparisons are correct
  - 3-pair: All 3 comparisons (A vs B, A vs C, B vs C) must be correct
  - 4-pair: All 6 comparisons must be correct

## 🔧 Available Scripts

### 1. 2-Pair Evaluation (`inference_2pair_hf.py`)

Evaluates models on 186 pairwise comparisons from 2-pair data.

```bash
python3 inference_2pair_hf.py \
    --model_name "qwen2.5-7b-instruct" \
    --template "pairwise_2pair" \
    --max_workers 32
```

**Expected Rankings**: A>B, B>A, A=B (3 classes)

### 2. 3-Pair Evaluation (`inference_3pair_hf.py`)

Evaluates models on 450 pairwise comparisons from 3-pair data, with group-level accuracy.

```bash
python3 inference_3pair_hf.py \
    --model_name "qwen2.5-7b-instruct" \
    --template "pairwise_2pair" \
    --max_workers 32
```

**Expected Rankings**: A>B, B>A, A>C, C>A, B>C, C>B (6 classes)
**Group Requirement**: All 3 comparisons in each group must be correct

### 3. 4-Pair Evaluation (`inference_4pair_hf.py`)

Evaluates models on 888 pairwise comparisons from 4-pair data, with group-level accuracy.

```bash
python3 inference_4pair_hf.py \
    --model_name "qwen2.5-7b-instruct" \
    --template "pairwise_2pair" \
    --max_workers 32
```

**Expected Rankings**: 12 possible pairwise combinations (12 classes)
**Group Requirement**: All 6 comparisons in each group must be correct

## 🤖 Supported Models

### Changing Models

You can easily switch between different models by modifying the `--model_name` parameter:

```bash
# Qwen2.5 models
python3 inference_2pair_hf.py --model_name "qwen2.5-7b-instruct"
python3 inference_2pair_hf.py --model_name "qwen2.5-vl-7b-instruct"

# Other supported models
python3 inference_2pair_hf.py --model_name "gpt-4o"
python3 inference_2pair_hf.py --model_name "gemini-2.0-flash"
python3 inference_2pair_hf.py --model_name "claude-3.5-sonnet"
```

### Model Requirements

- **Multimodal Capability**: Models must support both text and image inputs
- **API Access**: For API-based models (GPT-4, Claude), ensure you have valid API keys
- **Local Models**: For local models, ensure they're properly installed and accessible

### Model Configuration

Models are configured through the `genaibench.mllm_tools.MLLM_Models` class. To add a new model:

1. Check `genaibench/mllm_tools.py` for existing model configurations
2. Add your model configuration if needed
3. Use the model name in the evaluation scripts

## 📈 Results and Output

### Output Format

Results are saved as JSON files in the `results` directories:


### Key Metrics

- **Individual Accuracy**: Basic pairwise prediction accuracy
- **Group Accuracy**: For 3-pair/4-pair, stricter metric requiring all comparisons in a group to be correct
- **Processing Time**: Total evaluation time
- **Vote Distribution**: Breakdown of model predictions (A>B, B>A, A=B, etc.)

## ⚙️ Advanced Configuration

### Parallel Processing

```bash
# Use more workers for faster processing
python3 inference_2pair_hf.py --max_workers 64

# Use fewer workers for memory-constrained environments
python3 inference_2pair_hf.py --max_workers 8
```

### Resume Evaluation

```bash
# Resume from previous results (won't re-run completed evaluations)
python3 inference_2pair_hf.py --overwrite False

# Force re-run all evaluations
python3 inference_2pair_hf.py --overwrite True
```

### Testing Mode

```bash
# Quick test with limited samples
python3 inference_2pair_hf.py --max_examples 10

# Test specific subsets
python3 inference_3pair_hf.py --max_examples 5  # 5 groups = 15 samples
python3 inference_4pair_hf.py --max_examples 3  # 3 groups = 18 samples
```

## 📝 Prompt Templates

The evaluation uses optimized prompt templates located in `genaibench/templates/image_edition/`:

- **`pairwise_2pair.txt`**: Optimized for 3-class ranking (A>B, B>A, A=B)
- **`pairwise.txt`**: Original template with 4-class ranking

### Template Features

- Clear evaluation criteria
- Standardized output format `[[A>B]]`, `[[B>A]]`, `[[A=B]]`
- Comprehensive evaluation guidelines
- Consistent formatting across different model types

## 🔍 Understanding Results

### Example Output Analysis

```
🎯 Individual Accuracy: 0.5000 (93/186)
🎯 Group Accuracy: 0.3000 (45/150)

📊 Detailed Statistics:
   A>B: 78 (41.9%)
   B>A: 76 (40.9%) 
   A=B: 32 (17.2%)
```

This shows:
- Model correctly predicted 93 out of 186 pairwise comparisons (50% accuracy)
- For 3-pair groups, only 45 out of 150 groups had ALL comparisons correct (30% group accuracy)
- Model tends to predict A>B slightly more often than B>A

### Performance Interpretation

- **Individual Accuracy > 60%**: Good performance
- **Individual Accuracy > 70%**: Strong performance  
- **Individual Accuracy > 80%**: Excellent performance
- **Group Accuracy**: Much stricter metric, typically 20-40% lower than individual accuracy

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Check model availability
   python3 -c "from genaibench.mllm_tools import MLLM_Models; print(MLLM_Models('your-model-name'))"
   ```

2. **Memory Issues**
   ```bash
   # Reduce workers and batch size
   python3 inference_2pair_hf.py --max_workers 4
   ```

3. **API Rate Limits**
   ```bash
   # Reduce workers to avoid rate limiting
   python3 inference_2pair_hf.py --max_workers 8
   ```

4. **Dataset Loading Issues**
   ```bash
   # Test dataset access
   python3 -c "from datasets import load_dataset; ds = load_dataset('TIGER-Lab/EditReward-Bench'); print(len(ds['train']))"
   ```

## 📚 Citation

If you use this evaluation suite, please cite our paper:

```bibtex
@article{wu2025editreward,
  title={EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing},
  author={Wu, Keming and Jiang, Sicong and Ku, Max and Nie, Ping and Liu, Minghao and Chen, Wenhu},
  journal={arXiv preprint arXiv:2509.26346},
  year={2025}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your model configuration if needed
4. Test your changes thoroughly
5. Submit a pull request

## 📄 License

This evaluation suite is released under the MIT License.

---

**Keywords**: image editing, reward model, preference learning, pairwise comparison, multimodal, benchmark, evaluation
