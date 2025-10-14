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
# from .calc_accuracy import calc_accuracy_with_ties, calc_accuracy_without_ties

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
        # print(current_thresholds)
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
            # print(current_thresholds)
        # print("epsilon_star:", epsilon_star)
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
    # 读取数据
    with open(json_path, "r") as f:
        dataset = json.load(f)

    score_dict = defaultdict(list)

    for idx, item in enumerate(tqdm(dataset, desc="Evaluating ImagenHub")):
        prompt = item["instruction"]
        image_src = [item["input_path"]]       # 单张原图
        image_paths = [item["output_path"]]         # 单张生成图
        prompts = [prompt]

        # 模型打分
        with torch.no_grad():
            rewards = inferencer.reward(
                prompts=prompts,
                image_src=image_src,
                image_paths=image_paths
            )
        pred_score = rewards[0][0].item()   # 模型预测
        gt_score = item["score"]            # 人类评分
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

        # === 日志输出 ===
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
    # r_avg = sum(z_score_list) / len(z_score_list) if z_score_list else 0.0
    final_text = f"[ImagenHub] Average Spearman Correlation = {r_avg:.4f}\n"
    print(final_text)
    log_file.write(final_text)

    # 保存 JSON 文件
    spearman_results["average"] = r_avg
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(spearman_results, f, indent=2, ensure_ascii=False)

    print(f"Spearman results saved to {json_out_path}")
    return r_avg

def evaluate_imagenhub_from_json_v2(json_path, inferencer, log_file, json_out_path):
    """
    修改后的评估函数，按 prompt 分组，并使用两种更鲁棒的 Spearman 相关性计算方法。
    """
    # 1. 读取数据并按 prompt 进行分组
    with open(json_path, "r") as f:
        dataset = json.load(f)

    prompt_grouped_scores = defaultdict(list)
    print("Step 1/3: Running model inference and grouping by prompt...")
    for idx, item in enumerate(tqdm(dataset, desc="Evaluating ImagenHub")):
        prompt = item["instruction"]
        
        # 将数据聚合到 prompt_grouped_scores 字典中
        # 稍后统一进行模型推理，以提高效率
        prompt_grouped_scores[prompt].append({
            "gt_score": item["score"],
            "model_name": item.get("model", "unknown"),
            "image_src": [item["input_path"]],
            "image_paths": [item["output_path"]],
            "filename": item.get("filename", f"sample_{idx}")
        })

    # 2. 对每个 prompt 内的数据进行模型打分
    print("\nStep 2/3: Calculating scores for each prompt group...")
    for prompt, items in tqdm(prompt_grouped_scores.items(), desc="Scoring prompts"):
        for item in items:
            # 模型打分
            with torch.no_grad():
                rewards = inferencer.reward(
                    prompts=[prompt],
                    image_src=item["image_src"],
                    image_paths=item["image_paths"]
                )
            item["pred_score"] = rewards[0][0].item() # 将预测分数存回字典

            del rewards
            torch.cuda.empty_cache()
            gc.collect()

            # === 日志输出 (可选，但有助于调试) ===
            log_text = (
                f"\n--- Prompt: {prompt} | Model: {item['model_name']} ---\n"
                f"GT Score: {item['gt_score']:.4f}, Pred Score: {item['pred_score']:.4f}\n"
            )
            log_file.write(log_text)
            log_file.flush()

    # 3. 计算两种新的 Spearman 相关性指标
    print("\nStep 3/3: Calculating improved Spearman correlations...")
    per_prompt_spearmans = []
    all_gt_scores = []
    all_normalized_pred_scores = []

    for prompt, items in prompt_grouped_scores.items():
        if len(items) < 2:
            continue  # 无法计算相关性

        gt_list = [x['gt_score'] for x in items]
        pred_list = [x['pred_score'] for x in items]

        # === 计算指标1: 平均 Per-Prompt Spearman 相关性 ===
        r_prompt, _ = spearmanr(pred_list, gt_list)
        if not np.isnan(r_prompt):  # 如果标准差为0，spearmanr会返回nan
            per_prompt_spearmans.append(r_prompt)

        # === 计算指标2: 全局 Z-Score 归一化 Spearman 相关性 ===
        # 对当前 prompt 的 pred_list 进行 Z-Score 归一化
        mean_pred = np.mean(pred_list)
        std_pred = np.std(pred_list)
        
        if std_pred > 1e-6: # 避免除以零
            normalized_preds = [(s - mean_pred) / std_pred for s in pred_list]
        else:
            normalized_preds = [0.0] * len(pred_list)
        
        all_gt_scores.extend(gt_list)
        all_normalized_pred_scores.extend(normalized_preds)

    # --- 计算最终结果 ---
    # 指标1的最终值
    avg_per_prompt_spearman = np.mean(per_prompt_spearmans) if per_prompt_spearmans else 0.0
    
    # 指标2的最终值
    overall_zscore_spearman, _ = spearmanr(all_normalized_pred_scores, all_gt_scores)
    if np.isnan(overall_zscore_spearman):
        overall_zscore_spearman = 0.0

    # --- 结果汇总与保存 ---
    final_text = (
        f"\n====== [ImagenHub] Final Results ======\n"
        f"Total Prompts Evaluated: {len(per_prompt_spearmans)} / {len(prompt_grouped_scores)}\n"
        f"Metric 1: Average of Per-Prompt Spearman Correlation = {avg_per_prompt_spearman:.4f}\n"
        f"Metric 2: Overall Spearman Correlation (with Z-Score Norm) = {overall_zscore_spearman:.4f}\n"
    )
    print(final_text)
    log_file.write(final_text)
    
    # 保存 JSON 文件
    spearman_results = {
        "avg_per_prompt_spearman": avg_per_prompt_spearman,
        "overall_zscore_spearman": overall_zscore_spearman,
        "evaluated_prompt_count": len(per_prompt_spearmans),
        "total_prompt_count": len(prompt_grouped_scores)
    }
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(spearman_results, f, indent=2, ensure_ascii=False)

    print(f"Spearman results saved to {json_out_path}")
    
    # 返回一个主要指标，例如更鲁棒的 Z-Score Spearman
    return overall_zscore_spearman

def evaluate_aurora_from_json_pointwise(json_path, inferencer, log_file, json_out_path):
    # 读取数据
    with open(json_path, "r") as f:
        dataset = json.load(f)

    score_dict = defaultdict(list)

    for idx, item in enumerate(tqdm(dataset, desc="Evaluating Aurora-Pointwise")):
        prompt = item["prompt"]
        image_src = [item["input"]]       # 单张原图
        image_paths = [item["gen"]]         # 单张生成图
        prompts = [prompt]

        # 模型打分
        rewards = inferencer.reward(
            prompts=prompts,
            image_src=image_src,
            image_paths=image_paths
        )
        pred_score = rewards[0][0].item()   # 模型预测
        gt_score = item["score"]            # 人类评分
        model_name = item.get("model", "unknown")

        score_dict[model_name].append({
            "filename": item.get("filename", f"sample_{idx}"),
            "model": model_name,
            "gt": gt_score,
            "pred": pred_score,
        })

        # === 日志输出 ===
        log_text = (
            f"\n--- Aurora-Pointwise Sample {idx+1} ---\n"
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

        log_text = f"[Aurora-Pointwise] Model={model}, Spearman={r:.4f}\n"
        print(log_text)
        log_file.write(log_text)

        spearman_results[model] = r
        z_score_list.append(r)

    # 平均 Spearman
    r_avg = sum(z_score_list) / len(z_score_list) if z_score_list else 0.0
    final_text = f"[Aurora-Pointwise] Average Spearman Correlation = {r_avg:.4f}\n"
    print(final_text)
    log_file.write(final_text)

    # 保存 JSON 文件
    spearman_results["average"] = r_avg
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(spearman_results, f, indent=2, ensure_ascii=False)

    print(f"Spearman results saved to {json_out_path}")
    return r_avg

def get_pairwise_gt(human_vote, i, j):
    """
    REVISED: Determines ground truth for a pair (i, j) by parsing the preference string.
    - i, j: 0-based indices of the images being compared.
    - Maps indices to letters (0->'A', 1->'B', ...).
    - Parses preference strings like "A>B", "A=B=Good", "A>B>C".
    Returns: 1 if i > j, -1 if j > i, 0 for a tie.
    """
    pref_key = "ranking" # Assume this is the key, add fallbacks if needed
    if pref_key not in human_vote:
        return 0 # Cannot determine preference

    pref_str = human_vote[pref_key]

    # Handle 2-pair specific labels first
    if pref_str == "A>B":
        return 1
    if pref_str in ["B>A", "A<B"]:  # Now correctly handles both B>A and A<B
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
            return 1 # i is ranked higher than j
        else:
            return -1 # j is ranked higher than i
    except Exception:
        return 0

import numpy as np # Make sure to have numpy imported
from itertools import combinations
# ... (keep all your other functions like suff_stats, calc_accuracy_with_ties, etc.) ...
def get_gt_ranking(human_vote, num_images):
    """
    REVISED: Parses the human_vote string to get the ground-truth ranking.
    - For 2-pair ties ("A=B=Good", "A=B=Bad"), it now returns the special string "tie".
    - For strict rankings, it returns a tuple of indices, e.g., (2, 0, 1) for "C>A>B".
    - Returns None only if the ranking string is malformed.
    """
    pref_key = "ranking"
    if pref_key not in human_vote:
        return None

    pref_str = human_vote[pref_key]

    # --- MODIFIED SECTION ---
    # Handle 2-pair cases, returning "tie" for tie labels
    if pref_str in ["A=B=Good", "A=B=Bad"]:
        return "tie"
    if pref_str == "A>B": return (0, 1)
    if pref_str in ["B>A", "A<B"]: return (1, 0)
    # --- END MODIFIED SECTION ---
    
    # General N-pair ranking strings (e.g., "C>A>B")
    # This logic assumes N-pair data does not contain ties.
    if "=" in pref_str:
        return None # Malformed for strict N-pair ranking

    try:
        # Remove separators to get the character order, e.g., "CAB"
        char_ranking = pref_str.replace(">", "").replace("<", "")
        if len(char_ranking) != num_images:
            return None # Malformed ranking string

        # Convert character order to index order, e.g., ['C', 'A', 'B'] -> (2, 0, 1)
        index_ranking = tuple(ord(char) - ord('A') for char in char_ranking)
        return index_ranking
    except Exception:
        return None

def calc_accuracy_with_ties_for_edit_reward_bench(h, m):
    """
    MODIFIED: This function now returns both the max accuracy and the optimal epsilon.
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


def evaluate_edit_reward_bench(inferencer, log_file, json_out_path):
    """
    REVISED: Provides highly detailed, traceable logging for every data item.
    - Generates a summary JSON (_accuracy.json) and a detailed per-item log JSON (_detailed_log.json).
    - Text log also contains the full item-by-item breakdown.
    """
    # This path is for the detailed per-item log file
    detailed_log_path = json_out_path.replace("_accuracy.json", "_detailed_log.json")
    
    base_path = "/pfs/training-data/kemingwu/workspace/EditReward/HPSv3/data/dataset/edit_reward_bench/"
    files_to_process = {
        "2-Pair": os.path.join(base_path, "EditReward_Bench_2pair.json"),
        "3-Pair": os.path.join(base_path, "EditReward_Bench_3pair.json"),
        "4-Pair": os.path.join(base_path, "EditReward_Bench_4pair.json"),
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
                    "predicted_rank": int(ranks[path_idx]) + 1 # 1-based rank
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

# def evaluate_edit_reward_bench(inferencer, log_file, json_out_path):
#     """
#     REVISED: Strict Ranking Accuracy for 2-pair data now uses the dynamically
#     calculated optimal epsilon to determine model ties.
#     """
#     base_path = "/pfs/training-data/kemingwu/workspace/EditReward/HPSv3/data/dataset/edit_reward_bench/"
#     files_to_process = {
#         "2-Pair": os.path.join(base_path, "EditReward_Bench_2pair.json"),
#         "3-Pair": os.path.join(base_path, "EditReward_Bench_3pair.json"),
#         "4-Pair": os.path.join(base_path, "EditReward_Bench_4pair.json"),
#     }

#     # Initialize counters for overall results
#     overall_human_labels, overall_model_diffs = [], []
#     overall_strict_correct, overall_total_samples = 0, 0
#     results = {}

#     for name, json_path in files_to_process.items():
#         if not os.path.exists(json_path):
#             print(f"File not found for {name}: {json_path}. Skipping.")
#             continue

#         log_file.write(f"\n===== Evaluating Benchmark: {name} =====\n")
#         print(f"\n===== Evaluating Benchmark: {name} =====")
        
#         with open(json_path, "r") as f:
#             dataset = json.load(f)

#         # --- Stage 1: Collect all model scores and ground truth info ---
#         # We need to do this first to calculate the optimal epsilon for the entire file
        
#         file_human_labels, file_model_diffs = [], []
#         # Store per-sample info {gt_ranking, scores} for Stage 2
#         samples_for_strict_eval = []

#         for idx, item in enumerate(tqdm(dataset, desc=f"Stage 1/2: Scoring {name}")):
#             # ... (score calculation logic remains the same) ...
#             prompt, path_src, paths_generated = item["prompt"], item["path_src"], item["paths_generated"]
#             if not path_src or not all(paths_generated): continue
#             scores = []
#             for gen_path in paths_generated:
#                 with torch.no_grad():
#                     rewards = inferencer.reward(prompts=[prompt], image_src=[path_src], image_paths=[gen_path])
#                 scores.append(rewards[0][0].item())
#                 del rewards; torch.cuda.empty_cache(); gc.collect()

#             # Store info for strict eval
#             gt_ranking_or_tie = get_gt_ranking(item["human_vote"], len(paths_generated))
#             samples_for_strict_eval.append({"gt": gt_ranking_or_tie, "scores": scores})
            
#             # Generate pairwise comparisons for pairwise acc
#             for i, j in combinations(range(len(paths_generated)), 2):
#                 model_diff = scores[i] - scores[j]
#                 gt = get_pairwise_gt(item["human_vote"], i, j)
#                 file_human_labels.append(gt)
#                 file_model_diffs.append(model_diff)

#         # --- Stage 2: Calculate accuracies using the collected data ---
        
#         # Calculate pairwise accuracy and get optimal_epsilon if applicable
#         optimal_epsilon = 0.0 # Default for no-tie datasets
#         if '2-Pair' in name:
#             pairwise_acc, optimal_epsilon = calc_accuracy_with_ties_for_edit_reward_bench(file_human_labels, file_model_diffs)
#             pairwise_acc_type = "Pairwise Acc (with ties)"
#             log_file.write(f"\nOptimal Epsilon for 2-Pair data: {optimal_epsilon:.4f}\n")
#         else: # 3-Pair and 4-Pair datasets have no ties
#             pairwise_acc = calc_accuracy_without_ties(file_human_labels, file_model_diffs)
#             pairwise_acc_type = "Pairwise Acc (without ties)"

#         # Calculate strict accuracy using the optimal_epsilon
#         file_strict_correct, file_total_samples = 0, 0
#         for sample in tqdm(samples_for_strict_eval, desc=f"Stage 2/2: Strict Acc ({name})"):
#             file_total_samples += 1
#             gt, scores = sample["gt"], sample["scores"]
#             is_correct = False
#             if gt == "tie":
#                 score_diff = abs(scores[0] - scores[1])
#                 if score_diff <= optimal_epsilon: # Use the calculated epsilon
#                     is_correct = True
#             elif gt is not None:
#                 model_ranking = tuple(np.argsort(scores)[::-1])
#                 if gt == model_ranking:
#                     is_correct = True
#             if is_correct:
#                 file_strict_correct += 1

#         strict_acc = file_strict_correct / file_total_samples if file_total_samples > 0 else 0.0

#         # Store and log results for the current file
#         results[name] = { "pairwise_accuracy": pairwise_acc, "strict_ranking_accuracy": strict_acc }
#         result_text = (
#             f"\n--- Result for {name} ---\n"
#             f"{pairwise_acc_type}: {pairwise_acc:.4f}\n"
#             f"Strict Ranking Accuracy: {strict_acc:.4f} ({file_strict_correct}/{file_total_samples} samples)\n"
#         )
#         print(result_text)
#         log_file.write(result_text)

#         # Aggregate for overall results
#         overall_human_labels.extend(file_human_labels)
#         overall_model_diffs.extend(file_model_diffs)
#         overall_strict_correct += file_strict_correct
#         overall_total_samples += file_total_samples

#     # --- Final Overall Results ---
#     # Overall pairwise uses the combined data to find the best overall metric
#     overall_pairwise_acc, _ = calc_accuracy_with_ties(overall_human_labels, overall_model_diffs)
#     # Overall strict is the micro-average of correct samples over all samples
#     overall_strict_acc = overall_strict_correct / overall_total_samples if overall_total_samples > 0 else 0.0
    
#     results["overall"] = { "pairwise_accuracy": overall_pairwise_acc, "strict_ranking_accuracy": overall_strict_acc }
#     final_text = (
#         f"\n====== [EditReward_Bench] Final Overall Results ======\n"
#         f"Overall Pairwise Accuracy (with ties): {overall_pairwise_acc:.4f}\n"
#         f"Overall Strict Ranking Accuracy: {overall_strict_acc:.4f} ({overall_strict_correct}/{overall_total_samples} samples)\n"
#     )
#     print(final_text)
#     log_file.write(final_text)

#     with open(json_out_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)
#     print(f"EditReward_Bench results saved to {json_out_path}")
    
#     return results["overall"]["pairwise_accuracy"]

# ===== 配置部分 =====
def main(args):
    # ===== 配置部分 =====
    model_name = args.model_name
    config_path = args.config_path
    evaluate_benchmark = args.evaluate_benchmark
    checkpoint_step = args.checkpoint_step
    inference_mode = args.inference_mode
    reward_dim = args.reward_dim
    rm_head_type = args.rm_head_type
    if inference_mode not in ["pairwise_inference", "single_inference"]:
        raise ValueError(f"Unsupported pair_wise_inference: {inference_mode}")

    if "MiMo" in model_name:
        checkpoint_path = (
            f"/pfs/training-data/kemingwu/hf_cache/models/EditReward/"
            f"{model_name}/{model_name}/checkpoint-{checkpoint_step}"
        )
    else:
        checkpoint_path = (
            f"/pfs/training-data/kemingwu/hf_cache/models/EditReward/"
            f"{model_name}/EditReward-Qwen2/checkpoint-{checkpoint_step}"
        )

    if evaluate_benchmark == "genai_edit_bench":
        json_path = "/pfs/training-data/kemingwu/workspace/EditReward/HPSv3/data/dataset/valid_set2_full.json"
    elif evaluate_benchmark == "aurora_bench_pairwise":
        json_path = "/pfs/training-data/kemingwu/workspace/EditReward/HPSv3/data/dataset/valid_aurora_human_ratings_pairwise.json"
    elif evaluate_benchmark == "aurora_bench_pairwise_revise":
        json_path = "/pfs/training-data/kemingwu/workspace/EditReward/HPSv3/data/dataset/valid_human_ratings_pairwise_with_scores_revise_20250923.json"
    elif evaluate_benchmark == "aurora_bench_pointwise":
        json_path = "/pfs/training-data/kemingwu/workspace/EditReward/HPSv3/data/dataset/valid_aurora_human_ratings_pointwise.json"
        inference_mode = "single_inference"
    elif "imagenhub_bench" in evaluate_benchmark:
        json_path = "/pfs/training-data/kemingwu/workspace/EditReward/HPSv3/data/dataset/valid_imagenhub_processed.json"
        inference_mode = "single_inference"
    elif evaluate_benchmark == "edit_reward_bench":
        inference_mode = "single_inference"
    else:
        raise ValueError(f"Unsupported benchmark: {evaluate_benchmark}")

    # 初始化模型
    inferencer = EditRewardInferencer(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=args.device,
        reward_dim=args.reward_dim,
        rm_head_type=args.rm_head_type
    )

    log_path = (
        f"/pfs/training-data/kemingwu/workspace/EditReward/HPSv3/results/{evaluate_benchmark}/"
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
        print(f"日志已保存到 {log_path}")
        return
    elif evaluate_benchmark == "aurora_bench_pointwise":
        json_out_path = log_path.replace(".txt", "_spearman.json")
        r_avg = evaluate_aurora_from_json_pointwise(json_path, inferencer, log_file, json_out_path)
        log_file.close()
        print(f"Final Aurora Spearman Correlation = {r_avg:.4f}")
        print(f"日志已保存到 {log_path}")
        return
    elif evaluate_benchmark == "edit_reward_bench":
        json_out_path = log_path.replace(".txt", "_accuracy.json")
        overall_acc = evaluate_edit_reward_bench(inferencer, log_file, json_out_path)
        log_file.close()
        print(f"Final EditReward_Bench Overall Accuracy = {overall_acc:.4f}")
        print(f"日志已保存到 {log_path}")
        return


        # 读取数据
    with open(json_path, "r") as f:
        dataset = json.load(f)

    # 保存人类标签 & 模型分差
    human_labels, model_diffs = [], []

    for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
        prompt = item["prompt"]
        image_src = [item["path_src"], item["path_src"]]
        image_paths = [item["path1"], item["path2"]]
        prompts = [prompt, prompt]

        if inference_mode == "pairwise_inference":
            # 模型预测分数
            with torch.no_grad():
                rewards = inferencer.reward(prompts=prompts, image_src=image_src, image_paths=image_paths)
            scores = [reward[0].item() for reward in rewards]
            diff = scores[0] - scores[1]

            del rewards
            torch.cuda.empty_cache()
            gc.collect()
        elif inference_mode == "single_inference":
            # 模型预测分数
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

        # 构造日志信息
        log_text = (
            f"\n--- Sample {idx+1} ---\n"
            f"Prompt: {prompt}\n"
            f"Scores: path1={scores[0]:.4f}, path2={scores[1]:.4f}, diff={diff:.4f}\n"
            f"Ground Truth: {gt} (0=tie, 1=path1, -1=path2)\n"
        )
        print(log_text)
        log_file.write(log_text)
        log_file.flush()

    # ===== 计算最终准确率 =====
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

    print(f"日志已保存到 {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EditReward model on benchmarks")
    parser.add_argument("--model_name", type=str, required=True, help="模型名称，例如 EditReward-Qwen2.5-7B-VL_dim1_loss_uncertainty_data_4k_20250913")
    parser.add_argument("--evaluate_benchmark", type=str, required=True, choices=["genai_edit_bench", "aurora_bench_pairwise", "aurora_bench_pairwise_revise", "imagenhub_bench", "imagenhub_bench_v2", "aurora_bench_pointwise", "edit_reward_bench"], help="选择 benchmark")
    parser.add_argument("--checkpoint_step", type=int, required=True, help="checkpoint step，例如 5034")
    parser.add_argument("--config_path", type=str, required=True, help="config path")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备，默认 cuda")
    parser.add_argument("--reward_dim", type=str, default="dim1", help="reward 维度，默认 dim1")
    parser.add_argument("--inference_mode", type=str, default="pairwise_inference", choices=["pairwise_inference", "single_inference"], help="pairwise inference 或 single inference")
    parser.add_argument("--rm_head_type", type=str, default="ranknet_multi_head", choices=["ranknet_multi_head", "ranknet_single_head"], help="ranknet 多头或单头")
    args = parser.parse_args()

    main(args)