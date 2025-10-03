import torch
from torch.utils.data import Dataset, DataLoader
import random
import json
import os
from tqdm import tqdm

class PairwiseEditOriginalDataset(Dataset):
    def __init__(
        self,
        json_list,
        soft_label=False,
        confidence_threshold=None,
    ):
        self.samples = []
        for json_file in json_list:
            with open(json_file, "r") as f:
                data = json.load(f)
            self.samples.extend(data)

        self.soft_label = soft_label
        self.confidence_threshold = confidence_threshold

        if confidence_threshold is not None:
            new_samples = []
            for sample in tqdm(
                self.samples, desc="Filtering samples according to confidence threshold"
            ):
                if sample.get("confidence", float("inf")) >= confidence_threshold:
                    new_samples.append(sample)
            self.samples = new_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        while True:
            index = idx
            try:
                item = self.get_single_item(idx)
                if item is None:
                    # 如果当前 sample 无效，递归取下一个
                    return self.__getitem__((idx + 1) % len(self.samples))
                return item
            except Exception as e:
                print(f"Error processing sample at index {idx}: {e}")
                import traceback
                traceback.print_exc()
                index = random.randint(0, len(self.samples) - 1)
                if index == idx:
                    continue
                idx = index

    def get_single_item(self, idx):
        try:
            sample = self.samples[idx]

            # Load image paths
            image_src = sample.get("path_src")
            image_1 = sample.get("path1")
            image_2 = sample.get("path2")

            if not (image_src and image_1 and image_2):
                return None
            if not (os.path.exists(image_src) and os.path.exists(image_1) and os.path.exists(image_2)):
                return None

            text_1 = sample.get("prompt", "")
            text_2 = sample.get("prompt", "")

            # 自适应提取 dim 分数
            A_scores, B_scores = None, None
            A_scores_overall, B_scores_overall = None, None

            for k, v in sample.items():
                if k.startswith("model1_dim") and k.endswith("_score") and v is not None:
                    A_scores = v
                elif k.startswith("model2_dim") and k.endswith("_score") and v is not None:
                    B_scores = v

            for k, v in sample.items():
                if k.startswith("model1_overall") and k.endswith("_score") and v is not None:
                    A_scores_overall = v
                elif k.startswith("model2_overall") and k.endswith("_score") and v is not None:
                    B_scores_overall = v

            # Process Label
            if self.soft_label:
                choice_dist = sample.get("choice_dist", None)
                if not choice_dist or sum(choice_dist) <= 0:
                    return None
                choice_dist = sorted(choice_dist, reverse=True)
                label = torch.tensor(choice_dist[0]) / torch.sum(torch.tensor(choice_dist))
            else:
                label = torch.tensor(1).float()

            return {
                "image_src": image_src,
                "image_1": image_1,
                "image_2": image_2,
                "text_1": text_1,
                "text_2": text_2,
                "label": label,
                "confidence": sample.get("confidence", 1.0),
                "A_scores": A_scores,
                "B_scores": B_scores,
                "A_scores_overall": A_scores_overall,
                "B_scores_overall": B_scores_overall,
            }
        except Exception as e:
            # 打印 warning，方便定位
            print(f"[WARN] Skipping idx={idx}, reason: {e}")
            return None