"""pip install transformers>=4.35.2
"""
import os
import torch
import time
from PIL import Image
from typing import List
from transformers import AutoModel, AutoTokenizer
from transformers.image_utils import load_image
from transformers.utils import is_flash_attn_2_available
from .mllm_utils import merge_images

class MiniCPMV():
    support_multi_image = True
    max_input_images=4
    def __init__(self, model_path:str="openbmb/MiniCPM-Llama3-V-2_5") -> None:
        """Llava model wrapper

        Args:
            model_path (str): model name
        """
        
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        self.model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16, device_map='auto', _attn_implementation=attn_implementation).eval()
        self.tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)

        print(f"Using {attn_implementation} for attention implementation")

        
    def __call__(self, inputs: List[dict]) -> str:
        """
        Args:
            inputs (List[dict]): [
                {
                    "type": "image",
                    "content": "https://example.com/test_image_1.jpg"
                },
                {
                    "type": "image",
                    "content": "https://example.com/test_image_2.jpg"
                },
                {
                    "type": "text",
                    "content": "What is difference between two images?"
                }
            ]
            Supports any form of interleaved format of image and text.
        """
        if self.support_multi_image:
            content = []
            num_images = [x["type"] for x in inputs].count("image")
            
            idx = 0
            while idx < len(inputs):
                _input = inputs[idx]
                if _input["type"] == "image":
                    images = []
                    while inputs[idx]["type"] == "image":
                        if isinstance(inputs[idx]["content"], str):
                            image = load_image(inputs[idx]["content"])
                        elif isinstance(inputs[idx]["content"], Image.Image):
                            image = inputs[idx]["content"]
                        else:
                            raise ValueError("Invalid image input", inputs[idx]["content"], "should be str or PIL.Image.Image")
                        images.append(image)
                        idx += 1
                    if len(images) > 2:
                        merged_image = merge_images(images)
                        content.append(merged_image)
                    else:
                        content += images
                elif _input["type"] == "text":
                    content.append(_input["content"])
                    idx += 1
                else:
                    raise ValueError("Invalid input type", _input["type"])                    
            
            messages = [{"role": "user", "content": content}]
            
            res = self.model.chat(
                image=None,
                msgs=messages,
                tokenizer=self.tokenizer,
                sampling=False, # if sampling=False, beam_search will be used by default
            )
            return res
        else:
            raise NotImplementedError
        
if __name__ == "__main__":
    model = MiniCPMV()
    # 0 shot
    zero_shot_exs = [
        {
            "type": "image",
            "content": "https://example.com/test_image_1.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image?"
        }
    ]
    # 1 shot
    one_shot_exs = [
        {
            "type": "image",
            "content": "https://example.com/test_image_1.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image? A zebra."
        },
        {
            "type": "image",
            "content": "https://example.com/test_image_2.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image?"
        }
    ]
    # 2 shot
    two_shot_exs = [
        {
            "type": "image",
            "content": "https://example.com/test_image_1.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image? A zebra."
        },
        {
            "type": "image",
            "content": "https://example.com/test_image_2.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image? A black cat."
        },
        {
            "type": "image",
            "content": "https://hips.hearstapps.com/hmg-prod/images/rabbit-breeds-american-white-1553635287.jpg?crop=0.976xw:0.651xh;0.0242xw,0.291xh&resize=980:*"
        },
        {
            "type": "text",
            "content": "What is in the image?"
        }
    ]
    print("### 0 shot")
    print(model(zero_shot_exs))
    print("### 1 shot")
    print(model(one_shot_exs))
    print("### 2 shot")
    print(model(two_shot_exs))
    """
    Output: a tiger and a zebra
    ### 0 shot
    The image features a zebra grazing on grass in a field.
    ### 1 shot
    A zebra.
    ### 2 shot
    A black cat.
    """
    