from EditReward.model.qwen2_5_vl_trainer import Qwen2_5_VLRewardModelBT_MultiHead
from EditReward.model.qwen3_vl_trainer import Qwen3_VLRewardModelBT_MultiHead


def is_qwen3_vl(model_name_or_path: str) -> bool:
    if model_name_or_path is None:
        return False
    name = model_name_or_path.lower()
    return ("qwen3-vl" in name) or ("qwen3vl" in name)


def get_reward_model_cls(model_name_or_path: str):
    if is_qwen3_vl(model_name_or_path):
        return Qwen3_VLRewardModelBT_MultiHead
    return Qwen2_5_VLRewardModelBT_MultiHead

