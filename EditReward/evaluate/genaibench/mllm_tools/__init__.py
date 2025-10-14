MLLM_LIST = ["blip2", "instructblip", "llava", "openflamingo", "fuyu", "kosmos2", "qwenVL", "qwen2vl", "qwen2.5vl", "qwen2.5vl-3b", "qwen2.5-7b-instruct", "qwen2.5-32b-instruct", "cogvlm", "mfuyu", "mllava", "idefics2", "idefics1", "emu2", "otterimage", "ottervideo", "vila", "gpt4v", "gpt4o", "gpt5", "gpt5-eval2", "gpt-5-mini", "gemini", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro2", "gemini-2.5-pro3", "mantis", "videollava", "minicpmv", "phi35_vision", "llava_onevision", "mimovl"]
from functools import partial

def get_mfuyu(model_name: str):
    from .mfuyu_eval import MFuyu
    if model_name == "mantis-8b-fuyu":
        return MFuyu(model_id="TIGER-Lab/Mantis-8B-Fuyu")
    else:
        raise ValueError(f'Invalid model name: {model_name}')
    
def get_mllava(model_name: str):
    from .mllava_eval import MLlava
    if model_name == "mantis-7b-llava":
        return MLlava(model_path="TIGER-Lab/Mantis-llava-7b")
    elif model_name == "mantis-7b-bakllava":
        return MLlava(model_path="TIGER-Lab/Mantis-bakllava-7b")
    elif model_name == "mantis-8b-clip-llama3":
        return MLlava(model_path="TIGER-Lab/Mantis-8B-clip-llama3")
    elif model_name == "mantis-8b-siglip-llama3":
        return MLlava(model_path="TIGER-Lab/Mantis-8B-siglip-llama3")
    elif model_name == "mantis-8b-siglip-llama3-ablation1":
        return MLlava(model_path="Mantis-VL/llava_siglip_llama3_8b_finetune_ablation1_8192_lora")
    elif model_name == "mantis-8b-siglip-llama3-ablation2":
        return MLlava(model_path="Mantis-VL/llava_siglip_llama3_8b_finetune_ablation2_8192_lora")
    elif model_name == "mantis-8b-siglip-llama3-ablation3":
        return MLlava(model_path="Mantis-VL/llava_siglip_llama3_8b_finetune_ablation3_8192_lora")
    elif model_name == "mantis-8b-siglip-llama3-ablation4":
        return MLlava(model_path="Mantis-VL/llava_siglip_llama3_8b_finetune_ablation4_8192_lora")
    elif model_name == "mantis-8b-siglip-llama3-obelics-min":
        return MLlava(model_path="Mantis-VL/llava_siglip_llama3_8b_finetune_obelics_min_8192_lora")
    else:
        raise ValueError(f'Invalid model name: {model_name}')
    
def get_mantis_idefics(model_name:str):
    from .idefics2_eval import Idefics2
    if model_name == "mantis-8b-idefics2_8192" or model_name == "mantis-8b-idefics2":
        return Idefics2(model_path="Mantis-VL/mantis-8b-idefics2_8192")
    elif model_name == "mantis-8b-idefics2_8192_qlora":
        return Idefics2(model_path="Mantis-VL/mantis-8b-idefics2_8192_qlora")
    elif model_name == "mantis-8b-idefics2-data-ablation-1_8192_qlora":
        return Idefics2(model_path="Mantis-VL/mantis-8b-idefics2-data-ablation-1_8192_qlora")
    elif model_name == "mantis-8b-idefics2-data-ablation-2_8192_qlora":
        return Idefics2(model_path="Mantis-VL/mantis-8b-idefics2-data-ablation-2_8192_qlora")
    elif model_name == "mantis-8b-idefics2-data-ablation-3_8192_qlora":
        return Idefics2(model_path="Mantis-VL/mantis-8b-idefics2-data-ablation-3_8192_qlora")
    elif model_name == "mantis-8b-idefics2-data-ablation-4_8192_qlora":
        # same as mantis-8b-idefics2_8192_qlora
        return Idefics2(model_path="Mantis-VL/mantis-8b-idefics2_8192_qlora")
    else:
        raise ValueError(f'Invalid model name: {model_name}')
 
def MLLM_Models(model_name:str):
    if model_name == "blip2":
        from .blip_flant5_eval import BLIP_FLANT5
        return BLIP_FLANT5
    elif model_name == "instructblip":
        from .instructblip_eval import INSTRUCTBLIP_FLANT5
        return INSTRUCTBLIP_FLANT5
    elif model_name == "llava":
        from .llava_eval import Llava
        return Llava
    elif model_name == "llavanext":
        from .llava_next_eval import LlavaNext
        return LlavaNext
    elif "openflamingo" in model_name.lower():
        if model_name == "openflamingo-9b":
            model_id = "openflamingo/OpenFlamingo-9B-vitl-mpt7b"
            input_type = "pretrained"
        elif model_name == "openflamingo-3b":
            model_id = "openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct"
            input_type = "pretrained"
        elif model_name == "llava-9b-openflamingo":
            model_id = "Mantis-VL/llava-9b-openflamingo_2048"
            input_type = "chat"
        elif model_name == "mantis-9b-openflamingo":
            model_id = "Mantis-VL/mantis-9b-openflamingo_2048"
            input_type = "chat"
        else:
            raise ValueError(f'Invalid model name: {model_name}')
        from .openflamingo_eval import OpenFlamingo
        return partial(OpenFlamingo, model_id=model_id, input_type=input_type)
    elif model_name == "fuyu":
        from .fuyu_eval import Fuyu
        return Fuyu
    elif model_name == "kosmos2":
        from .kosmos2_eval import Kosmos2
        return Kosmos2
    elif model_name == "qwenVL":
        from .qwenVL_eval import QwenVL
        return QwenVL
    elif model_name == "qwen2vl":
        from .qwen2vl_eval import Qwen2VL
        return Qwen2VL
    elif model_name == "qwen2.5vl":
        from .qwen2_5vl_eval import Qwen2_5VL
        return Qwen2_5VL
    elif model_name == "qwen2.5vl-3b":
        from .qwen2_5vl_3b_eval import Qwen2_5VL_3B
        return Qwen2_5VL_3B
    elif model_name == "qwen2.5-7b-instruct":
        from .qwen2_5_7b_instruct_eval import Qwen2_5_7B_Instruct
        return Qwen2_5_7B_Instruct
    elif model_name == "qwen2.5-32b-instruct":
        from .qwen2_5_3b_instruct_eval import Qwen2_5_32B_Instruct
        return Qwen2_5_32B_Instruct
    elif model_name == "cogvlm":
        from .cogvlm_eval import CogVLM
        return CogVLM
    elif model_name == "idefics2":
        from .idefics2_eval import Idefics2
        return Idefics2
    elif model_name == "idefics1":
        from .idefics1_eval import Idefics1
        return Idefics1
    elif model_name == "emu2":
        from .emu2_eval import Emu2
        return Emu2
    elif model_name == "otterimage":
        from .otterimage_eval import OtterImage
        return OtterImage
    elif model_name == "ottervideo":
        from .ottervideo_eval import OtterVideo
        return OtterVideo
    elif model_name == "vila":
        from .vila_eval import VILA
        return VILA
    elif model_name == "minicpm-V-2.5":
        from .minicpmv_eval import MiniCPMV
        return MiniCPMV
    elif "videollava" in model_name.lower():
        if model_name == "videollava":
            model_id = 'LanguageBind/Video-LLaVA-7B-hf'
        elif model_name == "videollava-image":
            model_id = 'LanguageBind/Video-LLaVA-7B-hf'
        elif model_name == "videollava-video":
            model_id = 'LanguageBind/Video-LLaVA-7B-hf'
        else:
            raise ValueError(f'Invalid model name: {model_name}')
        from .videollava_eval import VideoLlava
        return partial(VideoLlava, model_path=model_id)
    elif model_name.lower().startswith("gpt4"):
        if model_name == "gpt4v":
            from .gpt4_eval import GPT4V
            return GPT4V
        elif model_name == "gpt4o":
            from .gpt4_eval import GPT4O
            return GPT4O
        elif model_name == "gpt4o-mini":
            from .gpt4_eval import GPT4OMini
            return GPT4OMini
        else:
            raise ValueError(f'Invalid model name: {model_name}')
    elif model_name == "gpt5":
        from .gpt5_eval import GPT5
        return GPT5
    elif model_name == "gpt5-eval2":
        from .gpt5_eval2 import GPT5Eval2
        return GPT5Eval2
    elif model_name == "gpt-5-mini":
        from .gpt5_mini_eval import GPT5Mini
        return GPT5Mini
    elif model_name == "gemini-2.5-pro":
        from .gemini_25_pro_eval import Gemini25Pro
        return Gemini25Pro
    elif model_name == "gemini-2.5-pro3":
        from .gemini_25_pro_parallel_eval import Gemini25ProParallel
        return Gemini25ProParallel
    elif model_name == "gemini-2.5-flash":
        from .gemini_25_flash_eval import Gemini25Flash
        return Gemini25Flash
    elif model_name == "gemini-2.0-flash":
        from .gemini_20_flash_eval import Gemini20Flash
        return Gemini20Flash
    elif model_name == "gemini-1.5-pro2":
        from .gemini_15_pro2_eval import Gemini15Pro2
        return Gemini15Pro2
    elif model_name.lower().startswith("gemini"):
        if model_name == "gemini-1.5-pro":
            from .gemini_eval import Gemini
            return partial(Gemini, model_name="gemini-1.5-pro-latest")
        elif model_name == "gemini-1.5-flash":
            from .gemini_eval import Gemini
            return partial(Gemini, model_name="gemini-1.5-flash-latest")
    elif model_name.lower().startswith("mantis"):
        if "fuyu" in model_name.lower():
            return partial(get_mfuyu, model_name=model_name)
        elif "idefics2" in model_name.lower():
            from .idefics2_eval import Idefics2
            return partial(get_mantis_idefics, model_name=model_name)
        elif "openflamingo" in model_name.lower():
            raise NotImplementedError
        else:
            return partial(get_mllava, model_name=model_name)
    elif model_name == "phi35_vision":
        from .phi35_vision_eval import Phi35Vision
        return Phi35Vision
    elif model_name == "llava_onevision":
        from .llava_onevision_eval import LlavaOnevision
        return LlavaOnevision
    elif model_name == "mimovl":
        from .mimovl_eval_ultra_fast import MiMoVLUltraFast
        return MiMoVLUltraFast
    else:
        raise ValueError(f'Invalid model name: {model_name}, must be one of {MLLM_LIST}')
    
