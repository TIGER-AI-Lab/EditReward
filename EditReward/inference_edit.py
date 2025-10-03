import os
from collections.abc import Mapping
import torch
import safetensors
from pathlib import Path

from .dataset.utils import process_vision_info
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
from .train_qwen2_5_edit import create_model_and_processor
from .dataset.prompts import (
    INSTRUCTION_EDIT_FOLLOWING,
    INSTRUCTION_EDIT_QUALITY,
    INSTRUCTION_EDIT_OVERALL,
    INSTRUCTION_EDIT_OVERALL_DETAILED,
)

_MODEL_CONFIG_PATH = Path(__file__).parent / "config/"


def _insert_adapter_name_into_state_dict(
    state_dict: dict[str, torch.Tensor], adapter_name: str, parameter_prefix: str
) -> dict[str, torch.Tensor]:
    """Remap the state_dict keys to fit the PEFT model by inserting the adapter name."""
    peft_model_state_dict = {}
    for key, val in state_dict.items():
        if parameter_prefix in key:
            suffix = key.split(parameter_prefix)[1]
            if "." in suffix:
                suffix_to_replace = ".".join(suffix.split(".")[1:])
                key = key.replace(
                    suffix_to_replace, f"{adapter_name}.{suffix_to_replace}"
                )
            else:
                key = f"{key}.{adapter_name}"
            peft_model_state_dict[key] = val
        else:
            peft_model_state_dict[key] = val
    return peft_model_state_dict


class EditRewardInferencer:
    def __init__(
        self,
        config_path=None,
        checkpoint_path=None,
        device="cuda",
        differentiable=False,
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
            differentiable=differentiable,
        )

        self.device = device
        self.use_special_tokens = model_config.use_special_tokens
        self.reward_dim = reward_dim
        self.rm_head_type = rm_head_type

        # === 加载 checkpoint ===
        full_ckpt = os.path.join(checkpoint_path, "model.pth")
        lora_ckpt = os.path.join(checkpoint_path, "adapter_model.safetensors")
        non_lora_ckpt = os.path.join(checkpoint_path, "non_lora_state_dict.pth")

        if os.path.exists(full_ckpt):
            model_state_dict = torch.load(full_ckpt, map_location="cpu")
            model.load_state_dict(model_state_dict)
        else:
            lora_state_dict = safetensors.torch.load_file(lora_ckpt)
            non_lora_state_dict = torch.load(non_lora_ckpt, map_location="cpu")
            lora_state_dict = _insert_adapter_name_into_state_dict(
                lora_state_dict, adapter_name="default", parameter_prefix="lora_"
            )
            model_state_dict = model.state_dict()
            # model_state_dict.update(non_lora_state_dict)
            # model_state_dict.update(lora_state_dict)
            model.load_state_dict(model_state_dict, strict=True)

        model.eval()
        self.model = model
        self.processor = processor
        self.model.to(self.device)
        self.data_config = data_config

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side="right"):
        """Pad the sequences to the maximum length."""
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

    def _prepare_input(self, data):
        """Convert inputs to tensors on device."""
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
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
                    base_prompt = INSTRUCTION_EDIT_OVERALL_DETAILED.format(
                        text_prompt=text
                    )
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
                            {"type": "image", "image": img, "min_pixels": max_pixels, "max_pixels": max_pixels},
                            {"type": "text", "text": final_text},
                        ],
                    }
                ]
                message_list.append(out_message)
            return message_list

        def _build_batch(prompts, image_src, image_paths, reward_dim):
            messages = _build_messages(prompts, image_src, image_paths, reward_dim)
            image_inputs, _ = process_vision_info(messages)
            batch = self.processor(
                text=self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ),
                images=image_inputs,
                padding=True,
                return_tensors="pt",
                videos_kwargs={"do_rescale": True},
            )
            batch = self._prepare_inputs(batch)
            return batch, image_inputs

        if self.rm_head_type == "ranknet_multi_head":
            batch_dim1, image_inputs_1 = _build_batch(prompts, image_src, image_paths, reward_dim="dim1")
            batch_dim2, image_inputs_2 = _build_batch(prompts, image_src, image_paths, reward_dim="dim2")
            return {
                "batch_dim1": batch_dim1,
                "batch_dim2": batch_dim2,
                "image_src": image_src,
                "image_paths": image_paths,
                "prompts": prompts,
                "image_inputs_dim1": image_inputs_1,
                "image_inputs_dim2": image_inputs_2,
            }
        else:
            batch, image_inputs = _build_batch(prompts, image_src, image_paths, reward_dim=self.reward_dim)
            return {
                "batch": batch,
                "image_src": image_src,
                "image_paths": image_paths,
                "prompts": prompts,
                "image_inputs": image_inputs,
            }

    def reward(self, prompts, image_src, image_paths):
        batch = self.prepare_batch(image_src, image_paths, prompts)
        rewards = self.model(return_dict=True, **batch)["logits"]
        return rewards


if __name__ == "__main__":
    checkpoint_path = "your/local/path/to/checkpoint"
    config_path = "config/EditReward-Qwen2.5-7B-VL.yaml"
    device = "cuda"

    inferencer = EditRewardInferencer(config_path, checkpoint_path, device=device)

    image_src = [
        "assets/examples/source_img_1.png",
        "assets/examples/source_img_1.png",
    ]

    image_paths = [
        "assets/examples/target_img_1.png",
        "assets/examples/target_img_2.png",
    ]

    prompts = [
        "Add a green bowl on the branch",
        "Add a green bowl on the branch",
    ]

    rewards = inferencer.reward(prompts, image_src, image_paths)
    print(rewards[0][0].item())
    print(rewards[1][0].item())