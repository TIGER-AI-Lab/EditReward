import fire
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genaibench.mllm_tools import MLLM_Models
from genaibench.utils import load_template
from datasets import load_dataset
from PIL import Image
import io

def run_pairwise_comparison_embedded(model, source_image, instruct_prompt, image_a, image_b, prompt_template):
    """Run a pairwise comparison between two embedded PIL images"""
    prompt = prompt_template
    
    # Replace text placeholders
    prompt = prompt.replace("<source_prompt>", "")  # Not available in 2pair data
    prompt = prompt.replace("<target_prompt>", "")   # Not available in 2pair data
    prompt = prompt.replace("<instruct_prompt>", instruct_prompt)
    
    # Build model inputs
    model_inputs = []
    
    # Split prompt and insert images at appropriate positions
    parts = prompt.split("<source_image>")
    model_inputs.append({"type": "text", "content": parts[0]})
    model_inputs.append({"type": "image", "content": source_image})
    
    remaining = parts[1]
    parts = remaining.split("<left_output_image>")
    model_inputs.append({"type": "text", "content": parts[0]})
    model_inputs.append({"type": "image", "content": image_a})
    
    remaining = parts[1]
    parts = remaining.split("<right_output_image>")
    model_inputs.append({"type": "text", "content": parts[0]})
    model_inputs.append({"type": "image", "content": image_b})
    model_inputs.append({"type": "text", "content": parts[1]})
    
    response = model(model_inputs)
    return response

def parse_pairwise_response(response):
    """Parse response and return A>B, B>A, A=B, or unknown"""
    import re
    
    # Try multiple patterns to extract model vote
    patterns = [
        r"\[\[([^\]]+)\]\]",
        r"\\boxed\{([^}]+)\}",
        r"\[([ABab][><=][ABab])\]",
        r"(?:Final verdict|verdict|answer):\s*([ABab][><=][ABab])(?:\s*$|\s*\.?\s*$)",
        r"\b([ABab]\s*[><=]\s*[ABab])\b(?=\s*(?:\.|$|,|\s))",
        r"(?:Model\s+)?([ABab])\s+is\s+better",
        r"both\s+(?:are\s+)?(?:equal|same|tie|tied)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            if len(match.groups()) > 0:
                model_vote_raw = match.group(1).strip().upper().replace(" ", "")
            else:
                model_vote_raw = match.group(0).strip().upper().replace(" ", "")
            
            # Normalize to standard format (only A>B, B>A, A=B)
            if "A>B" in model_vote_raw or "A > B" in model_vote_raw:
                return "A>B"
            elif "B>A" in model_vote_raw or "B > A" in model_vote_raw:
                return "B>A"
            elif "A=B" in model_vote_raw or "A = B" in model_vote_raw or "A==B" in model_vote_raw:
                return "A=B"
            break
    
    # Look for "A is better", "B is better" patterns
    better_match = re.search(r"\b([ABab])\s+is\s+better", response, re.IGNORECASE)
    if better_match:
        letter = better_match.group(1).upper()
        return f"{letter}>{'B' if letter == 'A' else 'A'}"
    
    # Look for tie/equal patterns
    tie_patterns = [
        r"both\s+(?:are\s+)?(?:equal|same|tie|tied|equivalent)",
        r"(?:they\s+are\s+)?(?:equal|same|tie|tied|equivalent)",
        r"no\s+difference",
        r"can't\s+decide",
        r"both\s+(?:look|are)\s+good",
        r"both\s+(?:look|are)\s+bad"
    ]
    
    for pattern in tie_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return "A=B"
    
    return "Unknown"

def evaluate_2pair_hf_example(model, example, prompt_template, example_idx):
    """Evaluate a single 2-pair example using Hugging Face dataset"""
    # Extract data from dataset row
    instruction = example["instruction"]
    source_image = example["source_image"]  # PIL Image
    candidate_1 = example["candidate_1"]    # PIL Image
    candidate_2 = example["candidate_2"]    # PIL Image
    human_vote = example["ranking"]         # "A>B", "B>A", "A=B"
    
    # Run pairwise comparison using embedded images
    response = run_pairwise_comparison_embedded(
        model, 
        source_image, 
        instruction, 
        candidate_1, 
        candidate_2, 
        prompt_template
    )
    
    if response is None:
        return None
    
    comparison_result = parse_pairwise_response(response)
    if comparison_result is None:
        return None
    
    # Determine if model prediction matches human vote
    is_correct = False
    if human_vote == comparison_result:
        is_correct = True
    
    return {
        "id": example["id"],
        "instruction": instruction,
        "human_vote": human_vote,
        "model_response": response,
        "model_vote": comparison_result,
        "is_correct": is_correct,
        "score_1": example.get("score_1"),
        "score_2": example.get("score_2"),
        "model_1": example.get("model_1"),
        "model_2": example.get("model_2"),
        "comparison_type": example.get("comparison_type")
    }

def process_single_hf_example(args):
    """Process a single example - designed for parallel execution"""
    example, model_class, prompt_template, example_idx = args
    
    # Create a new model instance for this thread
    model = model_class()
    
    try:
        result = evaluate_2pair_hf_example(model, example, prompt_template, example_idx)
        return result
    except Exception as e:
        print(f"Error processing example {example_idx}: {e}")
        return None

def main(
    dataset_name: str = "TIGER-Lab/EditReward-Bench",
    model_name: str = "qwen2.5-7b-instruct",
    template: str = "pairwise_2pair",
    results_dir: str = None,
    overwrite: bool = False,
    max_examples: int = None,
    max_workers: int = 64
):
    """
    Parallel evaluation of Qwen2.5-7B-Instruct on 2-pair data from Hugging Face
    
    Args:
        dataset_name: Hugging Face dataset name (default: TIGER-Lab/EditReward-Bench)
        model_name: Name of the  model to evaluate
        template: Template name (default: pairwise_2pair - optimized for 3-class ranking)
        results_dir: Directory to save results
        overwrite: Whether to overwrite existing results
        max_examples: Maximum number of examples to process (for testing)
        max_workers: Number of parallel workers (default: 64)
        
    Note:
        Expected rankings: A>B, B>A, A=B (3 classes only)
        Filters data to only include 2pair samples
    """
    
    print("="*60)
    print("EditReward-Bench 2-Pair Evaluation (Hugging Face)")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Template: {template}")
    
    # Initialize model class
    if model_name != "random":
        model_class = MLLM_Models(model_name)
    else:
        raise ValueError("Random mode not supported in parallel version")
    
    # Load dataset from Hugging Face
    print(f"\n📥 Loading dataset from Hugging Face...")
    try:
        dataset = load_dataset(dataset_name)
        all_data = [dataset['train'][i] for i in range(len(dataset['train']))]
        
        # Filter to only 2pair samples
        data = [sample for sample in all_data if sample['dataset'] == '2pair']
        print(f"✅ Loaded {len(all_data)} total samples, filtered to {len(data)} 2pair samples")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Verify data
    if not data:
        print("❌ No 2pair samples found in dataset")
        return
    
    sample = data[0]
    print(f"📋 Columns: {list(sample.keys())}")
    print(f"🖼️  Image types:")
    print(f"   source_image: {type(sample['source_image'])}")
    print(f"   candidate_1: {type(sample['candidate_1'])}")
    print(f"   candidate_2: {type(sample['candidate_2'])}")
    
    # Check ranking distribution
    from collections import Counter
    rankings = [sample['ranking'] for sample in data]
    ranking_counts = Counter(rankings)
    print(f"📊 Ranking distribution:")
    for ranking, count in sorted(ranking_counts.items()):
        print(f"   {ranking}: {count} ({count/len(data)*100:.1f}%)")
    
    if max_examples:
        data = data[:max_examples]
        print(f"🔢 Limited to {max_examples} examples for testing")
    
    # Setup results directory
    if results_dir is None:
        results_dir = Path(__file__).parent / "results_2pair_hf_qwen2_5_7b"
    else:
        results_dir = Path(results_dir)
    
    results_file = results_dir / f"{model_name}_{template}_2pair_hf.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check existing results
    existing_results = {}
    if results_file.exists() and not overwrite:
        with open(results_file, 'r') as f:
            existing_data = json.load(f)
            for i, result in enumerate(existing_data.get("results", [])):
                if result and "is_correct" in result:
                    existing_results[i] = result
        print(f"📂 Found {len(existing_results)} existing results")
    
    # Load prompt template
    prompt_template = load_template("image_edition", template)
    
    # Prepare examples for processing
    examples_to_process = []
    for i, example in enumerate(data):
        if i not in existing_results:
            examples_to_process.append((example, model_class, prompt_template, i))
    
    print(f"\n🚀 Processing {len(examples_to_process)} examples with {max_workers} workers...")
    print(f"📊 Total dataset size: {len(data)}")
    
    # Process in parallel
    start_time = time.time()
    all_results = [None] * len(data)  # Pre-allocate with correct indices
    
    # Add existing results
    for idx, result in existing_results.items():
        all_results[idx] = result
    
    correct_count = sum(1 for r in existing_results.values() if r.get("is_correct", False))
    
    # Process new examples
    if examples_to_process:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            for example, model_class, prompt_template, idx in examples_to_process:
                future = executor.submit(process_single_hf_example, (example, model_class, prompt_template, idx))
                future_to_idx[future] = idx
            
            # Collect results with progress bar
            with tqdm(total=len(future_to_idx), desc="Processing examples") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        if result is not None:
                            all_results[idx] = result
                            if result.get("is_correct", False):
                                correct_count += 1
                            
                            # Save intermediate results every 10 completions
                            if pbar.n % 10 == 0:
                                intermediate_data = {
                                    "model_name": model_name,
                                    "template": template,
                                    "dataset_name": dataset_name,
                                    "total_examples": len([r for r in all_results if r is not None]),
                                    "correct_examples": correct_count,
                                    "accuracy": correct_count / len([r for r in all_results if r is not None]) if any(all_results) else 0,
                                    "results": all_results,
                                    "progress": f"{pbar.n}/{len(future_to_idx)} completed"
                                }
                                
                                with open(results_file, 'w') as f:
                                    json.dump(intermediate_data, f, indent=2)
                        
                    except Exception as e:
                        print(f"Error processing example {idx}: {e}")
                    
                    pbar.update(1)
    
    end_time = time.time()
    print(f"\n⏱️  Completed in {end_time - start_time:.2f} seconds")
    
    # Filter out None results
    valid_results = [r for r in all_results if r is not None]
    
    # Final results
    accuracy = correct_count / len(valid_results) if valid_results else 0
    
    final_data = {
        "model_name": model_name,
        "template": template,
        "dataset_name": dataset_name,
        "total_examples": len(valid_results),
        "correct_examples": correct_count,
        "accuracy": accuracy,
        "processing_time": end_time - start_time,
        "max_workers": max_workers,
        "data_type": "huggingface_2pair",
        "results": all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    print(f"\n📄 Results saved to {results_file}")
    print(f"🎯 Accuracy: {accuracy:.4f} ({correct_count}/{len(valid_results)})")
    
    # Print detailed statistics
    print(f"\n📊 Detailed Statistics:")
    vote_counts = {}
    for result in valid_results:
        vote = result.get("model_vote", "Unknown")
        vote_counts[vote] = vote_counts.get(vote, 0) + 1
    
    for vote, count in sorted(vote_counts.items()):
        percentage = count / len(valid_results) * 100
        print(f"   {vote}: {count} ({percentage:.1f}%)")
    
    return final_data

if __name__ == "__main__":
    fire.Fire(main)
