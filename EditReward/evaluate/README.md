# EditReward-Bench Evaluation Guide  
**(2-Pair, 3-Pair, 4-Pair Evaluation Using Our Framework)**

This guide explains how to evaluate any multimodal LLM (GPT/Gemini/HuggingFace models) on EditReward-Bench using our `inference_2pair_hf.py` pipeline.

---

## 1. Overview

The evaluation pipeline:

1. Loads `TIGER-Lab/EditReward-Bench` from HuggingFace  
2. Builds a multimodal prompt (instruction + images)  
3. Calls your model through a wrapper under `mllm_tools/`  
4. Compares model preference (A > B / B > A / A = B) with human votes  
5. Reports accuracy + detailed JSON results  

Once your model wrapper is added, you can run:

```bash
python inference_2pair_hf.py --model_name <your_model>
```

Same process applies to 3-pair and 4-pair evaluation.

---

## 2. Add Your Model to `mllm_tools`

To evaluate a new model, simply:

### **A. If your model is GPT/Gemini (API-based multimodal LLMs)**  
👉 **Use these files as reference:**

- `gpt5_eval.py`  
- `gemini_25_flash_eval.py`

These show how to:  
- convert the input list of `{text, image}` into API format  
- send requests to the model  
- return the model’s final text output  

Copy one of these files, modify API calls.

---

### **B. If your model is a HuggingFace open-source MLLM**  
👉 **Use the Qwen implementations as reference:**

- `qwen2.5_eval.py`
- (or any other Qwen eval file under mllm_tools)

These examples show how to:  
- load image/text inputs  
- run the HF model in a multimodal forward pass  
- extract the output string needed for evaluation  

---

### **C. Register your model**

After creating your wrapper, open:

```
EditReward/evaluate/genaibench/mllm_tools/__init__.py
```

Add your model to the registry, refer to:

```python
def MLLM_Models(model_name: str):
    if model_name == "blip2":
        from .blip_flant5_eval import BLIP_FLANT5
        return BLIP_FLANT5

    elif model_name == "gpt5":
        from .gpt5_eval import GPT5_EvalModel
        return GPT5_EvalModel
```

Now it becomes selectable via:

```
--model_name your_model_name
```

---

## 3. Run the 2-Pair Evaluation

### Quick test (small batch)

```bash
python inference_2pair_hf.py   --model_name your_model_name   --dataset_name TIGER-Lab/EditReward-Bench   --template pairwise_2pair   --max_examples 50   --results_dir results_test   --max_workers 8
```

### Full evaluation

```bash
python inference_2pair_hf.py   --model_name your_model_name   --dataset_name TIGER-Lab/EditReward-Bench   --template pairwise_2pair   --results_dir results_full   --max_workers 32   --overwrite True
```

---

## 4. Output & Results

Results are saved in:

```
<results_dir>/<model_name>_pairwise_2pair_2pair_hf.json
```

The JSON file includes:

- total examples  
- correct predictions  
- accuracy  
- per-sample results including:
  - instruction  
  - human vote  
  - model output  
  - whether the model was correct  

Perfect for benchmarking and comparing multiple models.

---

## 5. 3-Pair & 4-Pair Evaluation

The process is identical:

```bash
python inference_3pair_hf.py --model_name your_model_name
python inference_4pair_hf.py --model_name your_model_name
```

Your wrapper works for all three benchmarks.

---

## 6. Summary

To evaluate any model:

1. **Write a wrapper**  
   - GPT/Gemini → Copy from `gpt5_eval.py` / `gemini_25_flash_eval.py`  
   - HuggingFace open-source → Copy and modify from Qwen eval files  
2. **Register it** in `mllm_tools/__init__.py`  
3. **Run the evaluation script**  
4. **Read accuracy + JSON results**

---

