import os
import sys
# Add project root to Python path (optional, for local development)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass, field
from typing import Optional, List, Union
import numpy as np
import pandas as pd
import torch
from EditReward.dataset.utils import process_vision_info
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from EditReward.dataset.prompts import (
    INSTRUCTION_EDIT_FOLLOWING,
    INSTRUCTION_EDIT_QUALITY,
    INSTRUCTION_EDIT_OVERALL,
    INSTRUCTION_EDIT_OVERALL_DETAILED,
    INSTRUCTION_debug
)

prompt_with_special_token = """
Please provide the overall ratings of this image: <|Reward|>

END
"""

prompt_without_special_token = """
Please provide the overall ratings of this image: 
"""

class QWen2_5_VLDataCollator:
    def __init__(
        self,
        processor,
        with_instruction=True,
        max_pixels=256 * 28 * 28,  # Default max pixels
        min_pixels=256 * 28 * 28,  # Default min pixels
        use_special_tokens=True,
        reward_dim="instruction",
        rm_head_type="single_head",   # 新增参数，控制 collator 行为
    ):
        self.processor = processor
        self.with_instruction = with_instruction
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.use_special_tokens = use_special_tokens
        self.reward_dim = reward_dim
        self.rm_head_type = rm_head_type

    def _clean_message(
        self,
        texts,
        images_src,
        images,
        max_pixels=256 * 28 * 28,
        min_pixels=256 * 28 * 28,
        with_instruction=True,
        use_special_tokens=True,
        reward_dim="dim1"
    ):
        """
        remove unnecessary keys from message(very very necessary)
        """
        # print(f"use reward dim {reward_dim}.\n")

        message_list = []
        for text, image_src, image in zip(texts, images_src, images):
            # 1. 选择主 prompt
            if reward_dim == "dim1":
                base_prompt = INSTRUCTION_EDIT_FOLLOWING.format(text_prompt=text)
            elif reward_dim == "dim2":
                base_prompt = INSTRUCTION_EDIT_QUALITY.format(text_prompt=text)
            elif reward_dim == "overall":
                base_prompt = INSTRUCTION_EDIT_OVERALL.format(text_prompt=text)
            elif reward_dim == "overall_detail":
                base_prompt = INSTRUCTION_EDIT_OVERALL_DETAILED.format(text_prompt=text)

            # 2. 拼接特殊 token
            if use_special_tokens:
                final_text = base_prompt + prompt_with_special_token
            else:
                final_text = base_prompt + prompt_without_special_token

            out_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_src,
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels,
                        },
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels,
                        },
                        {
                            "type": "text",
                            "text": (
                                final_text
                            ),
                        },
                    ],
                }
            ]

            message_list.append(out_message)

        return message_list

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side="right"):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ["right", "left"]
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask

        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == "right" else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(
            sequences, padding, "constant", self.processor.tokenizer.pad_token_id
        )
        attention_mask_padded = torch.nn.functional.pad(
            attention_mask, padding, "constant", 0
        )

        return sequences_padded, attention_mask_padded

    def __call__(self, inputs, with_instruction=True):
        """ Preprocess inputs to token sequences and return a batch """
        images_src, images_1, images_2, texts_1, texts_2 = [], [], [], [], []
        for batch in inputs:
            images_src.append(batch["image_src"])
            texts_1.append(batch["text_1"])
            texts_2.append(batch["text_2"])
            images_1.append(batch["image_1"])
            images_2.append(batch["image_2"])

        def _build_batch(texts_1, texts_2, images_src, images_1, images_2, reward_dim):
            # 构造 dimX 的 batch
            messages_batch_1 = self._clean_message(texts_1, images_src, images_1, reward_dim=reward_dim)
            messages_batch_2 = self._clean_message(texts_2, images_src, images_2, reward_dim=reward_dim)

            image_inputs_1, _ = process_vision_info(messages_batch_1)
            image_inputs_2, _ = process_vision_info(messages_batch_2)
            image_inputs_1 = [np.array(x) / 255.0 for x in image_inputs_1]
            image_inputs_2 = [np.array(x) / 255.0 for x in image_inputs_2]

            do_rescale = False
            batch_1 = self.processor(
                text=self.processor.apply_chat_template(messages_batch_1, tokenize=False, add_generation_prompt=True),
                images=image_inputs_1,
                videos=None,
                padding=True,
                return_tensors="pt",
                images_kwargs={"do_rescale": do_rescale},
            )
            batch_2 = self.processor(
                text=self.processor.apply_chat_template(messages_batch_2, tokenize=False, add_generation_prompt=True),
                images=image_inputs_2,
                videos=None,
                padding=True,
                return_tensors="pt",
                images_kwargs={"do_rescale": do_rescale},
            )

            max_len = max(batch_1["input_ids"].shape[1], batch_2["input_ids"].shape[1])
            batch_1["input_ids"], batch_1["attention_mask"] = self._pad_sequence(
                batch_1["input_ids"], batch_1["attention_mask"], max_len, "right"
            )
            batch_2["input_ids"], batch_2["attention_mask"] = self._pad_sequence(
                batch_2["input_ids"], batch_2["attention_mask"], max_len, "right"
            )
            return batch_1, batch_2, image_inputs_1, image_inputs_2

        # === 构造返回 ===
        if self.rm_head_type == "ranknet_multi_head" or self.rm_head_type == "ranknet_share_head" or self.rm_head_type == "ranknet_multi_head_regression":
            # 同时构造 dim1 / dim2
            batch_1_dim1, batch_2_dim1, image_inputs_1, image_inputs_2 = _build_batch(
                texts_1, texts_2, images_src, images_1, images_2, reward_dim="dim1"
            )
            batch_1_dim2, batch_2_dim2, _, _ = _build_batch(
                texts_1, texts_2, images_src, images_1, images_2, reward_dim="dim2"
            )
            A_scores = torch.stack([torch.tensor(x["A_scores"]) for x in inputs])
            B_scores = torch.stack([torch.tensor(x["B_scores"]) for x in inputs])
            try:
                A_scores_overall = torch.stack([
                    torch.tensor(x["A_scores_overall"]) for x in inputs
                ])
                B_scores_overall = torch.stack([
                    torch.tensor(x["B_scores_overall"]) for x in inputs
                ])
            except Exception as e:
                print("==== Debug Info ====")
                for i, x in enumerate(inputs):
                    print(i, x.get("A_scores_overall"), x.get("B_scores_overall"))
                raise e
            
            batch = {
                "batch_1_dim1": batch_1_dim1,
                "batch_2_dim1": batch_2_dim1,
                "batch_1_dim2": batch_1_dim2,
                "batch_2_dim2": batch_2_dim2,
                "text_1": texts_1,
                "text_2": texts_2,
                "image_src": images_src,
                "image_1": image_inputs_1,
                "image_2": image_inputs_2,
                "A_scores": A_scores,
                "B_scores": B_scores,
                "A_scores_overall": A_scores_overall,
                "B_scores_overall": B_scores_overall,
            }
        else:
            # 单头
            batch_1, batch_2, image_inputs_1, image_inputs_2 = _build_batch(
                texts_1, texts_2, images_src, images_1, images_2, reward_dim=self.reward_dim
            )
            A_scores = torch.stack([torch.tensor(x["A_scores"]) for x in inputs])
            B_scores = torch.stack([torch.tensor(x["B_scores"]) for x in inputs])

            batch = {
                "batch_1": batch_1,
                "batch_2": batch_2,
                "text_1": texts_1,
                "text_2": texts_2,
                "image_src": images_src,
                "image_1": image_inputs_1,
                "image_2": image_inputs_2,
                "A_scores": A_scores,
                "B_scores": B_scores,
            }

        return batch