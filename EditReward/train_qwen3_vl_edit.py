import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import fire
from dataclasses import asdict
from functools import partial
import torch

from EditReward.model.qwen3_vl_trainer import Qwen3_VLRewardModelBT_MultiHead
from EditReward.model.qwen2_5_vl_trainer import (
    VLMRewardTrainer,
    compute_multi_attr_accuracy,
    PartialEmbeddingUpdateCallback,
)
from EditReward.dataset.pairwise_edit_dataset import PairwiseEditOriginalDataset
from EditReward.dataset.data_collator_qwen3_edit import QWen3_VLDataCollator
from EditReward.utils.parser import ModelConfig, PEFTLoraConfig, TrainingConfig, DataConfig
from EditReward.utils.training_utils import load_model_from_checkpoint, find_target_linear_names
from EditReward.utils.parser import parse_args_with_yaml
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model
from trl import get_kbit_device_map, get_quantization_config

try:
    import flash_attn
except ImportError:
    flash_attn = None
    print("Flash Attention is not installed. Falling to SDPA.")


def get_model_embeddings_parameters(model):
    try:
        if hasattr(model, "get_input_embeddings"):
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "parameters"):
                return emb.parameters()
    except Exception as e:
        print("DEBUG: get_input_embeddings() failed:", e)

    for attr in ["embed_tokens", "word_embeddings", "token_embedding", "token_embed", "embeddings"]:
        if hasattr(model, attr):
            emb = getattr(model, attr)
            if hasattr(emb, "parameters"):
                return emb.parameters()
    return None


def safe_set_embedding_requires_grad(model, requires_grad=False):
    params_iter = get_model_embeddings_parameters(model)
    if params_iter is None:
        print("WARNING: Cannot find embedding layer on model; skip freezing embeddings.")
        return False
    try:
        for p in params_iter:
            p.requires_grad = requires_grad
        return True
    except Exception as e:
        print("WARNING: Failed to set requires_grad on embedding params:", e)
        return False


def create_model_and_processor(
    model_config,
    peft_lora_config,
    training_args,
    cache_dir=None,
):
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path, padding_side="right", cache_dir=cache_dir
    )

    special_token_ids = None
    if model_config.use_special_tokens:
        special_tokens = ["<|Reward|>"]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

    model = Qwen3_VLRewardModelBT_MultiHead.from_pretrained(
        model_config.model_name_or_path,
        output_dim=model_config.output_dim,
        reward_token=model_config.reward_token,
        special_token_ids=special_token_ids,
        torch_dtype=torch_dtype,
        attn_implementation=(
            "flash_attention_2"
            if not training_args.disable_flash_attn2 and flash_attn is not None
            else "sdpa"
        ),
        cache_dir=cache_dir,
        rm_head_type=model_config.rm_head_type,
        rm_head_kwargs=model_config.rm_head_kwargs,
        pooling_strategy=model_config.pooling_strategy,
        **model_kwargs,
    )

    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(processor.tokenizer))

    if training_args.bf16:
        model.to(torch.bfloat16)
    if training_args.fp16:
        model.to(torch.float16)

    if model.rm_head_type == "ranknet_multi_head" or model.rm_head_type == "ranknet_multi_head_regression":
        for h in model.rm_heads.values():
            h.to(torch.float32)
    else:
        model.rm_head.to(torch.float32)

    if peft_lora_config.lora_enable:
        target_modules = find_target_linear_names(
            model,
            num_lora_modules=peft_lora_config.num_lora_modules,
            lora_namespan_exclude=peft_lora_config.lora_namespan_exclude,
        )
        peft_config = LoraConfig(
            target_modules=target_modules,
            r=peft_lora_config.lora_r,
            lora_alpha=peft_lora_config.lora_alpha,
            lora_dropout=peft_lora_config.lora_dropout,
            task_type=peft_lora_config.lora_task_type,
            use_rslora=peft_lora_config.use_rslora,
            bias="none",
            modules_to_save=peft_lora_config.lora_modules_to_save,
        )
        model = get_peft_model(model, peft_config)
    else:
        peft_config = None

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    return model, processor, peft_config


def save_configs_to_json(data_config, training_args, model_config, peft_lora_config):
    config_dict = {
        "data_config": asdict(data_config),
        "training_args": asdict(training_args),
        "model_config": asdict(model_config),
        "peft_lora_config": asdict(peft_lora_config),
    }
    del config_dict["training_args"]["local_rank"]
    del config_dict["training_args"]["_n_gpu"]

    save_path = os.path.join(training_args.output_dir, "model_config.json")
    os.makedirs(training_args.output_dir, exist_ok=True)
    print(training_args.output_dir)
    with open(save_path, "w") as f:
        json.dump(config_dict, f, indent=4)


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def _get_visual_module(model_to_configure):
    visual = getattr(model_to_configure, "visual", None)
    if visual is None:
        visual = getattr(model_to_configure, "vision_tower", None)
    if visual is None and hasattr(model_to_configure, "model"):
        visual = getattr(model_to_configure.model, "visual", None)
        if visual is None:
            visual = getattr(model_to_configure.model, "vision_tower", None)
    return visual


def train(config, local_rank=0, debug=False):
    (data_config, training_args, model_config, peft_lora_config), config_path = (
        parse_args_with_yaml(
            (DataConfig, TrainingConfig, ModelConfig, PEFTLoraConfig), config, is_train=True
        )
    )
    training_args.output_dir = os.path.join(
        training_args.output_dir, config.split("/")[-1].split(".")[0]
    )
    training_args.logging_dir = training_args.output_dir

    assert not (
        peft_lora_config.lora_enable and model_config.freeze_llm
    ), "When using LoRA, the LLM should not be frozen. If you want to freeze the LLM, please disable LoRA."
    if not peft_lora_config.lora_enable:
        assert (
            not peft_lora_config.vision_lora
        ), "Error: model_config.lora_enable is not enabled, but model_config.vision_lora is enabled."
    else:
        if peft_lora_config.lora_namespan_exclude is None:
            peft_lora_config.lora_namespan_exclude = []
        if not peft_lora_config.vision_lora:
            peft_lora_config.lora_namespan_exclude += ["visual", "vision_tower"]

    model, processor, peft_config = create_model_and_processor(
        model_config=model_config,
        peft_lora_config=peft_lora_config,
        training_args=training_args,
    )

    if training_args.load_from_pretrained is not None:
        model, checkpoint_step = load_model_from_checkpoint(
            model,
            training_args.load_from_pretrained,
            training_args.load_from_pretrained_step,
        )
    model.train()

    if peft_lora_config.lora_enable:
        model_to_configure = model.model
    else:
        model_to_configure = model
        set_requires_grad(model_to_configure.model.parameters(), not model_config.freeze_llm)

        if not safe_set_embedding_requires_grad(model_to_configure.model, False):
            if hasattr(model_to_configure, "model") and model_to_configure is not model_to_configure.model:
                if safe_set_embedding_requires_grad(model_to_configure, False):
                    print("INFO: frozen embeddings on model_to_configure wrapper.")
                else:
                    print("INFO: Could not freeze embeddings on either model or wrapper.")

    if not peft_lora_config.vision_lora:
        visual = _get_visual_module(model_to_configure)
        if visual is None:
            print("WARNING: Cannot find `visual`/`vision_tower` on model; skip freezing vision tower.")
        else:
            set_requires_grad(visual.parameters(), not model_config.freeze_vision_tower)
            if hasattr(visual, "merger"):
                set_requires_grad(visual.merger.parameters(), model_config.tune_merger)
            elif model_config.tune_merger:
                print("WARNING: tune_merger=True but `merger` not found on vision tower; skip.")

    if model_config.trainable_visual_layers:
        visual = _get_visual_module(model_to_configure)
        blocks = getattr(visual, "blocks", None) if visual is not None else None
        if blocks is None and visual is not None:
            blocks = getattr(visual, "layers", None)
        if blocks is None:
            print("WARNING: trainable_visual_layers is set but cannot find vision blocks; skip per-layer freezing.")
        else:
            assert model_config.trainable_visual_layers <= len(
                blocks
            ), "trainable_visual_layers should be <= number of visual blocks"
            freeze_layer_num = (
                len(blocks) - model_config.trainable_visual_layers
                if model_config.trainable_visual_layers > 0
                else 0
            )
            for index, layer in enumerate(blocks):
                if index < freeze_layer_num:
                    set_requires_grad(layer.parameters(), False)
                else:
                    set_requires_grad(layer.parameters(), True)

    if (
        model_to_configure.rm_head_type == "ranknet_multi_head"
        or model_to_configure.rm_head_type == "ranknet_multi_head_regression"
    ):
        for h in model_to_configure.rm_heads.values():
            set_requires_grad(h.parameters(), True)
    else:
        set_requires_grad(model_to_configure.rm_head.parameters(), True)

    train_dataset = PairwiseEditOriginalDataset(
        data_config.train_json_list,
        data_config.soft_label,
        data_config.confidence_threshold,
    )

    test_set_dict = {}
    for item in data_config.test_json_list:
        test_set_dict[item[0]] = PairwiseEditOriginalDataset(
            item[1],
            data_config.soft_label,
            data_config.confidence_threshold,
        )

    print(f"===> Selected {len(train_dataset)} samples for training.")
    for key, value in test_set_dict.items():
        print(f"===> Selected {len(value)} samples for {key} testing.")

    num_gpu = int(os.environ.get("WORLD_SIZE", 1))
    data_collator = QWen3_VLDataCollator(
        processor,
        max_pixels=data_config.max_pixels,
        min_pixels=data_config.min_pixels,
        with_instruction=data_config.with_instruction,
        reward_dim=data_config.reward_dim,
        use_special_tokens=model_config.use_special_tokens,
        rm_head_type=model_config.rm_head_type,
    )
    compute_metrics = partial(compute_multi_attr_accuracy)

    actual_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * num_gpu
    )
    total_steps = training_args.num_train_epochs * len(train_dataset) // actual_batch_size
    if training_args.save_epochs is not None:
        training_args.save_steps = round(
            training_args.save_epochs * len(train_dataset) / actual_batch_size
        )
    if training_args.eval_epochs is not None:
        training_args.eval_steps = round(
            training_args.eval_epochs * len(train_dataset) / actual_batch_size
        )
    if training_args.logging_epochs is not None:
        training_args.logging_steps = round(
            training_args.logging_epochs * len(train_dataset) / actual_batch_size
        )

    if training_args.local_rank == -1 or training_args.local_rank == 0:
        print(f"===> Using {num_gpu} GPUs.")
        print(f"===> Total Batch Size: {actual_batch_size}")
        print(f"===> Training Epochs: {training_args.num_train_epochs}")
        print(f"===> Total Steps: {total_steps}")
        print(f"===> Save Steps: {training_args.save_steps}")
        print(f"===> Eval Steps: {training_args.eval_steps}")
        print(f"===> Logging Steps: {training_args.logging_steps}")

    if training_args.local_rank == -1 or training_args.local_rank == 0:
        save_configs_to_json(data_config, training_args, model_config, peft_lora_config)

    print(train_dataset)

    special_token_ids = model.special_token_ids
    callbacks = []
    if special_token_ids is not None:
        callbacks.append(PartialEmbeddingUpdateCallback(special_token_ids, processor.tokenizer))

    trainer = VLMRewardTrainer(
        model=model,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=(test_set_dict if training_args.conduct_eval else None),
        peft_config=peft_config,
        callbacks=callbacks,
        loss_type=model_config.loss_type,
        loss_hyperparameters=model_config.loss_hyperparameters,
        tied_threshold=data_config.tied_threshold,
        visualization_steps=training_args.visualization_steps,
        max_viz_samples=training_args.max_viz_samples,
        rm_head_type=model_config.rm_head_type,
    )
    trainer.train()

    if training_args.local_rank == -1 or training_args.local_rank == 0:
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, os.path.join(training_args.output_dir, "final_model.pth"))
        model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    fire.Fire(train)


