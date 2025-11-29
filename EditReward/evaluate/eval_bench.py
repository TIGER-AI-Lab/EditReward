import sys, os, json
from datetime import datetime
import torch, gc
from tqdm import tqdm
import argparse
import math
from scipy.stats import spearmanr
from collections import defaultdict
import numpy as np
from itertools import combinations

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from EditReward import EditRewardInferencer

def suff_stats(h, m, epsilon):
    """
    +------------+-----------+-----------+-----------+
    | Notation   |          Model Prediction         |
    |            |     <     |     =     |     >     |
    +------------+-----------+-----------+-----------+
    |  Human   < |     C     |     Tm    |     D     |
    |  Label   = |     Th    |    Thm    |     Th    |
    |          > |     D     |     Tm    |     C     |
    +------------+-----------+-----------+-----------+
    C: Consistent on the preference,
    D: Discordant on the preference,
    Th: Human ties but model doesn't,
    Tm: Model ties but human doesn't,
    Thm: Both human and model ties,
    epsilon: threshold for ties
    """
    C = D = Th = Tm = Thm = 0

    for hi, mi in zip(h, m):
        if hi == 0 and abs(mi) <= epsilon:
            Thm += 1
        elif hi == 0:
            Th += 1
        elif abs(mi) <= epsilon:
            Tm += 1
        elif hi * mi > 0:
            C += 1
        else:
            D += 1
    return C, D, Th, Tm, Thm

def calc_acc(C, D, Th, Tm, Thm):
    # This function calculates the current accuracy based on the statistics
    return (C + Thm) / (C + D + Th + Tm + Thm)

def calc_accuracy_with_ties(h, m):
    """
    algorithm: https://arxiv.org/abs/2305.14324
    O(N^2logN)
    Input:
        h: list of N human labels, 1 for prefer A, -1 for prefer B, 0 for ties
        m: list of N model predictions, can be obtained by Score(A) - Score(B)
    Output:
        acc_star: accuracy-with-ties
    """
    try:
        C, D, Th, Tm, Thm = suff_stats(h, m, -1)
        
        sorted_pairs = sorted(zip(h, m), key=lambda x: abs(x[1]))
        
        acc_star = float('-inf')
        epsilon_star = 0
        epsilon_curr = -1

        current_stat = {
            'C': C, 'D': D, 'Th': Th, 'Tm': Tm, 'Thm': Thm
        }
        for hi, mi in sorted_pairs:
            # update the statistics by removing the current pair
            if hi == 0 and abs(mi) < epsilon_curr:
                current_stat['Thm'] -= 1
            elif hi == 0:
                current_stat['Th'] -= 1
            elif abs(mi) < epsilon_curr:
                current_stat['Tm'] -= 1
            elif hi * mi > 0:
                current_stat['C'] -= 1
            else:
                current_stat['D'] -= 1

            # update the epsilon value
            epsilon_curr = abs(mi)

            # update the statistics by adding the current pair
            if hi == 0 and abs(mi) <= epsilon_curr:
                current_stat['Thm'] += 1
            elif hi == 0:
                current_stat['Th'] += 1
            elif abs(mi) <= epsilon_curr:
                current_stat['Tm'] += 1
            elif hi * mi > 0:
                current_stat['C'] += 1
            else:
                current_stat['D'] += 1

            # calculate the new tau value
            acc_curr = calc_acc(**current_stat)

            if acc_curr > acc_star:
                acc_star = acc_curr
                epsilon_star = epsilon_curr
        return acc_star
    except Exception as e:
        print("Error in tie_calibration:", e)
        return 0
    

def calc_accuracy_without_ties(h, m):
    """
    Input:
        h: list of N human labels, 1 for prefer A, -1 for prefer B, 0 for ties
        m: list of N model predictions, can be obtained by Score(A) - Score(B)
    Output:
        acc_star: accuracy-without-ties
    """
    C, D, Th, Tm, Thm = suff_stats(h, m, -1)
    return C / (C + D + Tm)

def evaluate_imagenhub_from_json(json_path, inferencer, log_file, json_out_path):
    # Load data
    with open(json_path, "r") as f:
        dataset = json.load(f)

    score_dict = defaultdict(list)

    for idx, item in enumerate(tqdm(dataset, desc="Evaluating ImagenHub")):
        prompt = item["instruction"]
        image_src = [item["input_path"]]
        image_paths = [item["output_path"]]
        prompts = [prompt]

        # Model scoring
        with torch.no_grad():
            rewards = inferencer.reward(
                prompts=prompts,
                image_src=image_src,
                image_paths=image_paths
            )
        pred_score = rewards[0][0].item()
        gt_score = item["score"]
        model_name = item.get("model", "unknown")

        score_dict[model_name].append({
            "filename": item.get("filename", f"sample_{idx}"),
            "model": model_name,
            "gt": gt_score,
            "pred": pred_score,
        })

        del rewards
        torch.cuda.empty_cache()
        gc.collect()

        # === Logging ===
        log_text = (
            f"\n--- ImagenHub Sample {idx+1} ---\n"
            f"Prompt: {prompt}\n"
            f"GT Score: {gt_score:.4f}, Pred Score: {pred_score:.4f}\n"
            f"Model: {model_name}\n"
        )
        print(log_text)
        log_file.write(log_text)
        log_file.flush()

    # ===== 计算相关性 =====
    spearman_results = {}
    z_score_list = []
    for model, gt_pred_list in score_dict.items():
        gt_list = [x["gt"] for x in gt_pred_list]
        pred_list = [x["pred"] for x in gt_pred_list]
        r, _ = spearmanr(pred_list, gt_list)

        log_text = f"[ImagenHub] Model={model}, Spearman={r:.4f}\n"
        print(log_text)
        log_file.write(log_text)

        spearman_results[model] = r
        z_score_list.append(r)

    # Averages a list of Fisher Z-transformed correlation scores and converts it back to a correlation coefficient
    z_avg = sum(z_score_list) / len(z_score_list)
    r_avg = (math.exp(2 * z_avg) - 1) / (math.exp(2 * z_avg) + 1)
    final_text = f"[ImagenHub] Average Spearman Correlation = {r_avg:.4f}\n"
    print(final_text)
    log_file.write(final_text)

    # Save JSON file
    spearman_results["average"] = r_avg
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(spearman_results, f, indent=2, ensure_ascii=False)

    print(f"Spearman results saved to {json_out_path}")
    return r_avg

def evaluate_imagenhub_from_json_v2(json_path, inferencer, log_file, json_out_path):
    """
    Improved evaluation function that groups by prompt and uses two more robust Spearman correlation calculation methods.
    """
    # 1. Load data and group by prompt
    with open(json_path, "r") as f:
        dataset = json.load(f)

    prompt_grouped_scores = defaultdict(list)
    print("Step 1/3: Running model inference and grouping by prompt...")
    for idx, item in enumerate(tqdm(dataset, desc="Evaluating ImagenHub")):
        prompt = item["instruction"]
        
        # Aggregate data into prompt_grouped_scores dictionary
        # Will perform model inference later for efficiency
        prompt_grouped_scores[prompt].append({
            "gt_score": item["score"],
            "model_name": item.get("model", "unknown"),
            "image_src": [item["input_path"]],
            "image_paths": [item["output_path"]],
            "filename": item.get("filename", f"sample_{idx}")
        })

    # 2. Score each prompt group
    print("\nStep 2/3: Calculating scores for each prompt group...")
    for prompt, items in tqdm(prompt_grouped_scores.items(), desc="Scoring prompts"):
        for item in items:
            # Model scoring
            with torch.no_grad():
                rewards = inferencer.reward(
                    prompts=[prompt],
                    image_src=item["image_src"],
                    image_paths=item["image_paths"]
                )
            item["pred_score"] = rewards[0][0].item()

            del rewards
            torch.cuda.empty_cache()
            gc.collect()

            # === Logging (optional, but helpful for debugging) ===
            log_text = (
                f"\n--- Prompt: {prompt} | Model: {item['model_name']} ---\n"
                f"GT Score: {item['gt_score']:.4f}, Pred Score: {item['pred_score']:.4f}\n"
            )
            log_file.write(log_text)
            log_file.flush()

    # 3. Calculate two improved Spearman correlation metrics
    print("\nStep 3/3: Calculating improved Spearman correlations...")
    per_prompt_spearmans = []
    all_gt_scores = []
    all_normalized_pred_scores = []

    for prompt, items in prompt_grouped_scores.items():
        if len(items) < 2:
            continue

        gt_list = [x['gt_score'] for x in items]
        pred_list = [x['pred_score'] for x in items]

        # === Metric 1: Average Per-Prompt Spearman Correlation ===
        r_prompt, _ = spearmanr(pred_list, gt_list)
        if not np.isnan(r_prompt):
            per_prompt_spearmans.append(r_prompt)

        # === Metric 2: Global Z-Score Normalized Spearman Correlation ===
        # Z-Score normalize pred_list for current prompt
        mean_pred = np.mean(pred_list)
        std_pred = np.std(pred_list)
        
        if std_pred > 1e-6:
            normalized_preds = [(s - mean_pred) / std_pred for s in pred_list]
        else:
            normalized_preds = [0.0] * len(pred_list)
        
        all_gt_scores.extend(gt_list)
        all_normalized_pred_scores.extend(normalized_preds)

    # --- Calculate final results ---
    # Final value for metric 1
    avg_per_prompt_spearman = np.mean(per_prompt_spearmans) if per_prompt_spearmans else 0.0
    
    # Final value for metric 2
    overall_zscore_spearman, _ = spearmanr(all_normalized_pred_scores, all_gt_scores)
    if np.isnan(overall_zscore_spearman):
        overall_zscore_spearman = 0.0

    # --- Aggregate and save results ---
    final_text = (
        f"\n====== [ImagenHub] Final Results ======\n"
        f"Total Prompts Evaluated: {len(per_prompt_spearmans)} / {len(prompt_grouped_scores)}\n"
        f"Metric 1: Average of Per-Prompt Spearman Correlation = {avg_per_prompt_spearman:.4f}\n"
        f"Metric 2: Overall Spearman Correlation (with Z-Score Norm) = {overall_zscore_spearman:.4f}\n"
    )
    print(final_text)
    log_file.write(final_text)
    
    # Save JSON file
    spearman_results = {
        "avg_per_prompt_spearman": avg_per_prompt_spearman,
        "overall_zscore_spearman": overall_zscore_spearman,
        "evaluated_prompt_count": len(per_prompt_spearmans),
        "total_prompt_count": len(prompt_grouped_scores)
    }
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(spearman_results, f, indent=2, ensure_ascii=False)

    print(f"Spearman results saved to {json_out_path}")
    
    # Return a primary metric, e.g., the more robust Z-Score Spearman
    return overall_zscore_spearman

def evaluate_aurora_from_json_pointwise(json_path, inferencer, log_file, json_out_path):
    # Load data
    with open(json_path, "r") as f:
        dataset = json.load(f)

    score_dict = defaultdict(list)

    for idx, item in enumerate(tqdm(dataset, desc="Evaluating Aurora-Pointwise")):
        prompt = item["prompt"]
        image_src = [item["input"]]
        image_paths = [item["gen"]]
        prompts = [prompt]

        # Model scoring
        rewards = inferencer.reward(
            prompts=prompts,
            image_src=image_src,
            image_paths=image_paths
        )
        pred_score = rewards[0][0].item()
        gt_score = item["score"]
        model_name = item.get("model", "unknown")

        score_dict[model_name].append({
            "filename": item.get("filename", f"sample_{idx}"),
            "model": model_name,
            "gt": gt_score,
            "pred": pred_score,
        })

        # === Logging ===
        log_text = (
            f"\n--- Aurora-Pointwise Sample {idx+1} ---\n"
            f"Prompt: {prompt}\n"
            f"GT Score: {gt_score:.4f}, Pred Score: {pred_score:.4f}\n"
            f"Model: {model_name}\n"
        )
        print(log_text)
        log_file.write(log_text)
        log_file.flush()

    # ===== Calculate correlation =====
    spearman_results = {}
    z_score_list = []
    for model, gt_pred_list in score_dict.items():
        gt_list = [x["gt"] for x in gt_pred_list]
        pred_list = [x["pred"] for x in gt_pred_list]
        r, _ = spearmanr(pred_list, gt_list)

        log_text = f"[Aurora-Pointwise] Model={model}, Spearman={r:.4f}\n"
        print(log_text)
        log_file.write(log_text)

        spearman_results[model] = r
        z_score_list.append(r)

    # Average Spearman
    r_avg = sum(z_score_list) / len(z_score_list) if z_score_list else 0.0
    final_text = f"[Aurora-Pointwise] Average Spearman Correlation = {r_avg:.4f}\n"
    print(final_text)
    log_file.write(final_text)

    # Save JSON file
    spearman_results["average"] = r_avg
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(spearman_results, f, indent=2, ensure_ascii=False)

    print(f"Spearman results saved to {json_out_path}")
    return r_avg

def get_pairwise_gt(human_vote, i, j):
    """
    Determines ground truth for a pair (i, j) by parsing the preference string.
    - i, j: 0-based indices of the images being compared.
    - Maps indices to letters (0->'A', 1->'B', ...).
    - Parses preference strings like "A>B", "A=B=Good", "A>B>C".
    Returns: 1 if i > j, -1 if j > i, 0 for a tie.
    """
    pref_key = "ranking"
    if pref_key not in human_vote:
        return 0

    pref_str = human_vote[pref_key]

    # Handle 2-pair specific labels first
    if pref_str == "A>B":
        return 1
    if pref_str in ["B>A", "A<B"]:
        return -1
    if pref_str in ["A=B=Good", "A=B=Bad"]:
        return 0

    # General N-pair logic for ranked strings like "A>B>C"
    try:
        char_i = chr(ord('A') + i)
        char_j = chr(ord('A') + j)

        pos_i = pref_str.find(char_i)
        pos_j = pref_str.find(char_j)

        if pos_i == -1 or pos_j == -1:
            return 0
        
        if pos_i < pos_j:
            return 1
        else:
            return -1
    except Exception:
        return 0

def get_gt_ranking(human_vote, num_images):
    """
    Parses the human_vote string to get the ground-truth ranking.
    - For 2-pair ties ("A=B=Good", "A=B=Bad"), it returns the special string "tie".
    - For strict rankings, it returns a tuple of indices, e.g., (2, 0, 1) for "C>A>B".
    - Returns None only if the ranking string is malformed.
    """
    pref_key = "ranking"
    if pref_key not in human_vote:
        return None

    pref_str = human_vote[pref_key]

    # Handle 2-pair cases, returning "tie" for tie labels
    if pref_str in ["A=B=Good", "A=B=Bad"]:
        return "tie"
    if pref_str == "A>B":
        return (0, 1)
    if pref_str in ["B>A", "A<B"]:
        return (1, 0)
    
    # General N-pair ranking strings (e.g., "C>A>B")
    # This logic assumes N-pair data does not contain ties.
    if "=" in pref_str:
        return None

    try:
        # Remove separators to get the character order, e.g., "CAB"
        char_ranking = pref_str.replace(">", "").replace("<", "")
        if len(char_ranking) != num_images:
            return None

        # Convert character order to index order, e.g., ['C', 'A', 'B'] -> (2, 0, 1)
        index_ranking = tuple(ord(char) - ord('A') for char in char_ranking)
        return index_ranking
    except Exception:
        return None

def calc_accuracy_with_ties_for_edit_reward_bench(h, m):
    """
    This function returns both the max accuracy and the optimal epsilon.
    """
    try:
        if not h: return 0.0, 0.0
        
        # Sort pairs by model score difference to iterate through possible epsilons
        unique_diffs = sorted(list(set([abs(s) for s in m])))
        thresholds = [0.0] + unique_diffs
        
        best_acc = -1.0
        best_epsilon = -1.0

        for epsilon in thresholds:
            C, D, Th, Tm, Thm = suff_stats(h, m, epsilon)
            acc = calc_acc(C, D, Th, Tm, Thm)
            if acc > best_acc:
                best_acc = acc
                best_epsilon = epsilon
        
        return best_acc, best_epsilon
    except Exception as e:
        print(f"Error in calc_accuracy_with_ties: {e}")
        return 0.0, 0.0


def evaluate_edit_reward_bench(inferencer, log_file, json_out_path, benchmark_data_dir=None):
    """
    Provides highly detailed, traceable logging for every data item.
    - Generates a summary JSON (_accuracy.json) and a detailed per-item log JSON (_detailed_log.json).
    - Text log also contains the full item-by-item breakdown.
    
    Args:
        inferencer: EditRewardInferencer instance
        log_file: File handle for logging
        json_out_path: Path to save output JSON
        benchmark_data_dir: Directory containing benchmark JSON files (optional)
    """
    detailed_log_path = json_out_path.replace("_accuracy.json", "_detailed_log.json")
    
    # Use provided directory or default relative path
    if benchmark_data_dir is None:
        benchmark_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "data", "dataset", "edit_reward_bench")
    
    files_to_process = {
        "2-Pair": os.path.join(benchmark_data_dir, "EditReward_Bench_2pair.json"),
        "3-Pair": os.path.join(benchmark_data_dir, "EditReward_Bench_3pair.json"),
        "4-Pair": os.path.join(benchmark_data_dir, "EditReward_Bench_4pair.json"),
    }

    # Initialize overall counters and a list for all detailed logs
    overall_human_labels, overall_model_diffs = [], []
    overall_strict_correct, overall_total_samples = 0, 0
    results = {}
    all_detailed_logs = [] # This will store the detailed log for every single item

    for name, json_path in files_to_process.items():
        if not os.path.exists(json_path):
            print(f"File not found for {name}: {json_path}. Skipping.")
            continue

        log_file.write(f"\n===== Evaluating Benchmark: {name} =====\n")
        print(f"\n===== Evaluating Benchmark: {name} =====")
        
        with open(json_path, "r") as f:
            dataset = json.load(f)

        file_human_labels, file_model_diffs = [], []
        samples_for_strict_eval = []

        for idx, item in enumerate(tqdm(dataset, desc=f"Stage 1/2: Scoring {name}")):
            prompt, path_src, paths_generated = item["prompt"], item["path_src"], item["paths_generated"]
            if not path_src or not all(paths_generated): continue
            
            scores = []
            for gen_path in paths_generated:
                with torch.no_grad():
                    rewards = inferencer.reward(prompts=[prompt], image_src=[path_src], image_paths=[gen_path])
                scores.append(rewards[0][0].item())
                del rewards; torch.cuda.empty_cache(); gc.collect()

            # Store all necessary info for Stage 2 processing and logging
            samples_for_strict_eval.append({
                "gt_ranking": get_gt_ranking(item["human_vote"], len(paths_generated)),
                "scores": scores,
                "prompt": prompt,
                "path_src": path_src,
                "paths_generated": paths_generated,
                "human_vote_str": item.get("human_vote", {}).get("preference", "N/A")
            })
            
            for i, j in combinations(range(len(paths_generated)), 2):
                model_diff = scores[i] - scores[j]
                gt = get_pairwise_gt(item["human_vote"], i, j)
                file_human_labels.append(gt)
                file_model_diffs.append(model_diff)

        # Calculate pairwise accuracy and get optimal_epsilon
        optimal_epsilon = 0.0
        if '2-Pair' in name:
            pairwise_acc, optimal_epsilon = calc_accuracy_with_ties_for_edit_reward_bench(file_human_labels, file_model_diffs)
            pairwise_acc_type = "Pairwise Acc (with ties)"
            log_file.write(f"\nOptimal Epsilon for 2-Pair data: {optimal_epsilon:.4f}\n")
        else:
            pairwise_acc = calc_accuracy_without_ties(file_human_labels, file_model_diffs)
            pairwise_acc_type = "Pairwise Acc (without ties)"

        # Calculate strict accuracy and generate detailed logs
        log_file.write("\n--- Detailed Item-by-Item Evaluation ---\n")
        file_strict_correct, file_total_samples = 0, 0
        for i, sample in enumerate(tqdm(samples_for_strict_eval, desc=f"Stage 2/2: Strict Acc ({name})")):
            file_total_samples += 1
            gt, scores = sample["gt_ranking"], sample["scores"]
            is_correct = False
            
            if gt == "tie":
                if abs(scores[0] - scores[1]) <= optimal_epsilon:
                    is_correct = True
            elif gt is not None:
                model_ranking_indices = tuple(np.argsort(scores)[::-1])
                if gt == model_ranking_indices:
                    is_correct = True
            
            if is_correct:
                file_strict_correct += 1

            # --- DETAILED LOGGING ---
            # Create a detailed dictionary for this item
            detailed_item_log = {
                "file_name": name,
                "sample_index": i,
                "prompt": sample["prompt"],
                "ground_truth": sample["human_vote_str"],
                "strict_accuracy_correct": is_correct,
                "model_outputs": []
            }
            # Populate model outputs with path, score, and rank
            model_ranking_indices = np.argsort(scores)[::-1]
            ranks = np.empty_like(model_ranking_indices)
            ranks[model_ranking_indices] = np.arange(len(scores))
            for path_idx, path in enumerate(sample["paths_generated"]):
                detailed_item_log["model_outputs"].append({
                    "path": path,
                    "score": scores[path_idx],
                    "predicted_rank": int(ranks[path_idx]) + 1
                })
            
            all_detailed_logs.append(detailed_item_log)
            log_file.write(f"\nSample {i+1}:\n" + json.dumps(detailed_item_log, indent=2) + "\n")
        
        # Store summary results for this file
        strict_acc = file_strict_correct / file_total_samples if file_total_samples > 0 else 0.0
        results[name] = { 
            "pairwise_accuracy": pairwise_acc, "strict_ranking_accuracy": strict_acc,
            "strict_correct_count": file_strict_correct, "strict_total_samples": file_total_samples,
            "pair_count": len(file_human_labels)
        }
        result_text = (
            f"\n--- Result for {name} ---\n"
            f"{pairwise_acc_type}: {pairwise_acc:.4f}\n"
            f"Strict Ranking Accuracy: {strict_acc:.4f} ({file_strict_correct}/{file_total_samples} samples)\n"
        )
        print(result_text)
        log_file.write(result_text)

        overall_human_labels.extend(file_human_labels)
        overall_model_diffs.extend(file_model_diffs)
        overall_strict_correct += file_strict_correct
        overall_total_samples += file_total_samples

    # --- Final Overall Results ---
    overall_pairwise_acc, _ = calc_accuracy_with_ties_for_edit_reward_bench(overall_human_labels, overall_model_diffs)
    overall_strict_acc = overall_strict_correct / overall_total_samples if overall_total_samples > 0 else 0.0
    
    results["overall"] = { 
        "pairwise_accuracy": overall_pairwise_acc, "strict_ranking_accuracy": overall_strict_acc,
        "strict_correct_count": overall_strict_correct, "strict_total_samples": overall_total_samples,
        "total_pair_count": len(overall_human_labels)
    }
    final_text = (
        f"\n====== [EditReward_Bench] Final Overall Results ======\n"
        f"Overall Pairwise Accuracy (with ties): {overall_pairwise_acc:.4f}\n"
        f"Overall Strict Ranking Accuracy: {overall_strict_acc:.4f} ({overall_strict_correct}/{overall_total_samples} samples)\n"
    )
    print(final_text)
    log_file.write(final_text)

    # --- SAVE BOTH JSON FILES ---
    # 1. Save the summary results file
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Summary results saved to {json_out_path}")

    # 2. Save the detailed per-item log file
    with open(detailed_log_path, "w", encoding="utf-8") as f:
        json.dump(all_detailed_logs, f, indent=2, ensure_ascii=False)
    print(f"Detailed per-item logs saved to {detailed_log_path}")
    
    return results["overall"]["pairwise_accuracy"]


def main(args):
    model_name = args.model_name
    config_path = args.config_path
    evaluate_benchmark = args.evaluate_benchmark
    checkpoint_step = args.checkpoint_step
    inference_mode = args.inference_mode
    reward_dim = args.reward_dim
    rm_head_type = args.rm_head_type
    
    if inference_mode not in ["pairwise_inference", "single_inference"]:
        raise ValueError(f"Unsupported inference_mode: {inference_mode}")

    # Construct checkpoint path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        # Default path structure
        if "MiMo" in model_name or "Qwen3-VL" in model_name:
            checkpoint_path = os.path.join(
                args.model_base_dir or "models",
                model_name,
                model_name,
                f"checkpoint-{checkpoint_step}"
            )
        else:
            checkpoint_path = os.path.join(
                args.model_base_dir or "models",
                model_name,
                "EditReward-Qwen2",
                f"checkpoint-{checkpoint_step}"
            )

    # Get benchmark data paths
    if args.benchmark_data_dir:
        benchmark_data_dir = args.benchmark_data_dir
    else:
        benchmark_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "data", "dataset")
    
    if evaluate_benchmark == "genai_edit_bench":
        json_path = args.benchmark_json_path or os.path.join(benchmark_data_dir, "valid_set2_full.json")
    elif evaluate_benchmark == "aurora_bench_pairwise":
        json_path = args.benchmark_json_path or os.path.join(benchmark_data_dir, "valid_aurora_human_ratings_pairwise.json")
    elif evaluate_benchmark == "aurora_bench_pairwise_revise":
        json_path = args.benchmark_json_path or os.path.join(benchmark_data_dir, "valid_human_ratings_pairwise_with_scores_revise.json")
    elif evaluate_benchmark == "aurora_bench_pointwise":
        json_path = args.benchmark_json_path or os.path.join(benchmark_data_dir, "valid_aurora_human_ratings_pointwise.json")
        inference_mode = "single_inference"
    elif "imagenhub_bench" in evaluate_benchmark:
        json_path = args.benchmark_json_path or os.path.join(benchmark_data_dir, "valid_imagenhub_processed.json")
        inference_mode = "single_inference"
    elif evaluate_benchmark == "edit_reward_bench":
        inference_mode = "single_inference"
        edit_reward_bench_dir = args.benchmark_data_dir or os.path.join(benchmark_data_dir, "edit_reward_bench")
    else:
        raise ValueError(f"Unsupported benchmark: {evaluate_benchmark}")

    # Initialize model
    inferencer = EditRewardInferencer(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=args.device,
        reward_dim=args.reward_dim,
        rm_head_type=args.rm_head_type
    )

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  "results", evaluate_benchmark)
    os.makedirs(output_dir, exist_ok=True)
    
    log_path = os.path.join(
        output_dir,
        f"{model_name}_step_{checkpoint_step}_{evaluate_benchmark}_inference_mode_{inference_mode}_f_{rm_head_type}_eval_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    log_file = open(log_path, "w", encoding="utf-8")

    if "imagenhub_bench" in evaluate_benchmark:
        json_out_path = log_path.replace(".txt", "_spearman.json")
        if evaluate_benchmark == "imagenhub_bench":
            r_avg = evaluate_imagenhub_from_json(json_path, inferencer, log_file, json_out_path)
        elif evaluate_benchmark == "imagenhub_bench_v2":
            r_avg = evaluate_imagenhub_from_json_v2(json_path, inferencer, log_file, json_out_path)
        log_file.close()
        print(f"Final ImagenHub Spearman Correlation = {r_avg:.4f}")
        print(f"Log saved to {log_path}")
        return
    elif evaluate_benchmark == "aurora_bench_pointwise":
        json_out_path = log_path.replace(".txt", "_spearman.json")
        r_avg = evaluate_aurora_from_json_pointwise(json_path, inferencer, log_file, json_out_path)
        log_file.close()
        print(f"Final Aurora Spearman Correlation = {r_avg:.4f}")
        print(f"Log saved to {log_path}")
        return
    elif evaluate_benchmark == "edit_reward_bench":
        json_out_path = log_path.replace(".txt", "_accuracy.json")
        overall_acc = evaluate_edit_reward_bench(inferencer, log_file, json_out_path, edit_reward_bench_dir)
        log_file.close()
        print(f"Final EditReward_Bench Overall Accuracy = {overall_acc:.4f}")
        print(f"Log saved to {log_path}")
        return

    # Load data
    with open(json_path, "r") as f:
        dataset = json.load(f)

    # Store human labels & model score differences
    human_labels, model_diffs = [], []

    for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
        prompt = item["prompt"]
        image_src = [item["path_src"], item["path_src"]]
        image_paths = [item["path1"], item["path2"]]
        prompts = [prompt, prompt]

        if inference_mode == "pairwise_inference":
            # Model prediction scores
            with torch.no_grad():
                rewards = inferencer.reward(prompts=prompts, image_src=image_src, image_paths=image_paths)
            scores = [reward[0].item() for reward in rewards]
            diff = scores[0] - scores[1]

            del rewards
            torch.cuda.empty_cache()
            gc.collect()
        elif inference_mode == "single_inference":
            # Model prediction scores
            with torch.no_grad():
                rewards_A = inferencer.reward(prompts=[prompts[0]], image_src=[image_src[0]], image_paths=[image_paths[0]])
                rewards_B = inferencer.reward(prompts=[prompts[1]], image_src=[image_src[1]], image_paths=[image_paths[1]])
            scores_A = [reward[0].item() for reward in rewards_A]
            scores_B = [reward[0].item() for reward in rewards_B]
            diff = scores_A[0] - scores_B[0]
            scores = [scores_A[0], scores_B[0]]
            del rewards_A, rewards_B
            torch.cuda.empty_cache()
            gc.collect()

        # Ground truth (1= prefer path1, -1= prefer path2, 0= tie)
        if item["model1_dim1_score"] == item["model2_dim1_score"]:
            gt = 0
        elif item["model1_dim1_score"] > item["model2_dim1_score"]:
            gt = 1
        else:
            gt = -1

        human_labels.append(gt)
        model_diffs.append(diff)

        # Construct log message
        log_text = (
            f"\n--- Sample {idx+1} ---\n"
            f"Prompt: {prompt}\n"
            f"Scores: path1={scores[0]:.4f}, path2={scores[1]:.4f}, diff={diff:.4f}\n"
            f"Ground Truth: {gt} (0=tie, 1=path1, -1=path2)\n"
        )
        print(log_text)
        log_file.write(log_text)
        log_file.flush()

    # ===== Calculate final accuracy =====
    acc_with_ties = calc_accuracy_with_ties(human_labels, model_diffs)
    acc_without_ties = calc_accuracy_without_ties(human_labels, model_diffs)

    final_text = (
        "\n====== Final Result ======\n"
        f"Samples: {len(human_labels)}\n"
        f"Accuracy with ties (paper metric): {acc_with_ties:.4f}\n"
        f"Accuracy without ties (hard pref only): {acc_without_ties:.4f}\n"
    )

    print(final_text)
    log_file.write(final_text)
    log_file.close()

    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EditReward model on benchmarks")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Model name, e.g., EditReward-Qwen2.5-7B-VL_dim1_loss_uncertainty_data_4k_20250913")
    parser.add_argument("--evaluate_benchmark", type=str, required=True, 
                        choices=["genai_edit_bench", "aurora_bench_pairwise", "aurora_bench_pairwise_revise", 
                                "imagenhub_bench", "imagenhub_bench_v2", "aurora_bench_pointwise", "edit_reward_bench"], 
                        help="Benchmark to evaluate")
    parser.add_argument("--checkpoint_step", type=int, required=True, 
                        help="Checkpoint step, e.g., 5034")
    parser.add_argument("--config_path", type=str, required=True, 
                        help="Path to model config YAML file")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Full path to checkpoint directory. If not provided, will construct from model_name and checkpoint_step")
    parser.add_argument("--model_base_dir", type=str, default=None,
                        help="Base directory for model checkpoints. Default: 'models'")
    parser.add_argument("--benchmark_data_dir", type=str, default=None,
                        help="Directory containing benchmark data files. If not provided, uses default relative path")
    parser.add_argument("--benchmark_json_path", type=str, default=None,
                        help="Path to specific benchmark JSON file. Overrides benchmark_data_dir for specific benchmarks")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results. Default: results/{benchmark_name}")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run on (default: cuda)")
    parser.add_argument("--reward_dim", type=str, default="dim1", 
                        help="Reward dimension (default: dim1)")
    parser.add_argument("--inference_mode", type=str, default="pairwise_inference", 
                        choices=["pairwise_inference", "single_inference"], 
                        help="Inference mode: pairwise_inference or single_inference")
    parser.add_argument("--rm_head_type", type=str, default="ranknet_multi_head", 
                        choices=["ranknet_multi_head", "ranknet_single_head"], 
                        help="RankNet head type: multi-head or single-head")
    args = parser.parse_args()

    main(args)