from EditReward.dataset.data_collator_qwen_edit import QWen2_5_VLDataCollator
from EditReward.dataset.data_collator_qwen3_edit import QWen3_VLDataCollator
from EditReward.model.qwen_vl_trainer import is_qwen3_vl


def get_vl_data_collator(
    processor,
    model_name_or_path: str,
    max_pixels,
    min_pixels,
    with_instruction,
    reward_dim,
    use_special_tokens,
    rm_head_type,
):
    if is_qwen3_vl(model_name_or_path):
        return QWen3_VLDataCollator(
            processor,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            with_instruction=with_instruction,
            reward_dim=reward_dim,
            use_special_tokens=use_special_tokens,
            rm_head_type=rm_head_type,
        )
    return QWen2_5_VLDataCollator(
        processor,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        with_instruction=with_instruction,
        reward_dim=reward_dim,
        use_special_tokens=use_special_tokens,
        rm_head_type=rm_head_type,
    )


