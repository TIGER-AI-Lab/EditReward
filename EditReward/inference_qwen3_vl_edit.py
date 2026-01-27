import os
from collections.abc import Mapping
from pathlib import Path

import torch
import safetensors

from .dataset.data_collator_qwen_edit import (
    prompt_with_special_token,
    prompt_without_special_token,
)
from .utils.parser import (
    ModelConfig,
    PEFTLoraConfig,
    TrainingConfig,
    DataConfig,
    parse_args_with_yaml,
)
from .train_qwen3_vl_edit import create_model_and_processor
from .dataset.prompts import (
    INSTRUCTION_EDIT_FOLLOWING,
    INSTRUCTION_EDIT_QUALITY,
    INSTRUCTION_EDIT_OVERALL,
    INSTRUCTION_EDIT_OVERALL_DETAILED,
)

_MODEL_CONFIG_PATH = Path(__file__).parent / "config/"


class EditRewardQwen3Inferencer:
    def __init__(
        self,
        config_path=None,
        checkpoint_path=None,
        device="cuda",
        reward_dim="dim1",
        rm_head_type="ranknet_multi_head",
    ):
        if config_path is None:
            config_path = os.path.join(_MODEL_CONFIG_PATH, config_path)

        print(f"config_path: {config_path}\n")
        (data_config, training_args, model_config, peft_lora_config), config_path = (
            parse_args_with_yaml(
                (DataConfig, TrainingConfig, ModelConfig, PEFTLoraConfig),
                config_path,
                is_train=False,
            )
        )
        training_args.output_dir = os.path.join(
            training_args.output_dir, config_path.split("/")[-1].split(".")[0]
        )

        model, processor, _ = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
            cache_dir=None,
        )

        self.device = device
        self.use_special_tokens = model_config.use_special_tokens
        self.reward_dim = reward_dim
        self.rm_head_type = rm_head_type

        full_ckpt = os.path.join(checkpoint_path, "model.pth")
        full_ckpt_safetensors = os.path.join(checkpoint_path, "model.safetensors")

        if os.path.exists(full_ckpt):
            state_dict = torch.load(full_ckpt, map_location="cpu")
        elif os.path.exists(full_ckpt_safetensors):
            import safetensors.torch
            state_dict = safetensors.torch.load_file(full_ckpt_safetensors, device="cpu")
        else:
            raise ValueError(f"Checkpoint not found at {checkpoint_path}")

        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        self.model = model
        self.processor = processor
        self.model.to(self.device)
        self.data_config = data_config

    def _prepare_input(self, data):
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        if isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device)
        return data

    def _prepare_inputs(self, inputs):
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs

    def prepare_batch(self, image_src, image_paths, prompts):
        max_pixels = 256 * 28 * 28
        min_pixels = 256 * 28 * 28

        def _build_messages(prompts, image_src, image_paths, reward_dim):
            message_list = []
            for text, src, img in zip(prompts, image_src, image_paths):
                if reward_dim == "dim1":
                    base_prompt = INSTRUCTION_EDIT_FOLLOWING.format(text_prompt=text)
                elif reward_dim == "dim2":
                    base_prompt = INSTRUCTION_EDIT_QUALITY.format(text_prompt=text)
                elif reward_dim == "overall":
                    base_prompt = INSTRUCTION_EDIT_OVERALL.format(text_prompt=text)
                elif reward_dim == "overall_detail":
                    base_prompt = INSTRUCTION_EDIT_OVERALL_DETAILED.format(text_prompt=text)
                else:
                    raise ValueError(f"Unknown reward_dim: {reward_dim}")

                final_text = (
                    base_prompt + prompt_with_special_token
                    if self.use_special_tokens
                    else base_prompt + prompt_without_special_token
                )

                out_message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": src, "min_pixels": min_pixels, "max_pixels": max_pixels},
                            {"type": "image", "image": img, "min_pixels": min_pixels, "max_pixels": max_pixels},
                            {"type": "text", "text": final_text},
                        ],
                    }
                ]
                message_list.append(out_message)
            return message_list

        def _build_batch(prompts, image_src, image_paths, reward_dim):
            messages = _build_messages(prompts, image_src, image_paths, reward_dim)
            flat_messages = [m[0] for m in messages]
            batch = self.processor.apply_chat_template(
                flat_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            batch = self._prepare_inputs(batch)
            return batch

        if self.rm_head_type == "ranknet_multi_head":
            batch_dim1 = _build_batch(prompts, image_src, image_paths, reward_dim="dim1")
            batch_dim2 = _build_batch(prompts, image_src, image_paths, reward_dim="dim2")
            return {"batch_dim1": batch_dim1, "batch_dim2": batch_dim2}

        batch = _build_batch(prompts, image_src, image_paths, reward_dim=self.reward_dim)
        return {"batch": batch}

    def reward(self, prompts, image_src, image_paths):
        batch = self.prepare_batch(image_src, image_paths, prompts)
        if "batch" in batch:
            rewards = self.model(return_dict=True, **batch["batch"])["logits"]
        else:
            rewards = self.model(
                return_dict=True,
                batch_dim1=batch["batch_dim1"],
                batch_dim2=batch["batch_dim2"],
            )["logits"]
        return rewards


