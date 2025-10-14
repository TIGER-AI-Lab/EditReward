import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import List, Dict, Any
import requests
from io import BytesIO


class Qwen2VL:
    support_multi_image = True
    merged_image_files = []  # For cleanup

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-7B-Instruct") -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto"
        ).eval()

    def __call__(self, inputs: List[dict]) -> str:
        if self.support_multi_image:
            messages = self.prepare_messages(inputs)
            response = self.get_parsed_output(messages)
            return response
        else:

            raise NotImplementedError

    def prepare_messages(self, inputs: List[dict]) -> List[dict]:
        """
        Prepare messages in Qwen2-VL format
        """
        messages = []
        current_message = {"role": "user", "content": []}

        for item in inputs:
            if item["type"] == "image":
                if isinstance(item["content"], str):
                    if item["content"].startswith("http"):
                        response = requests.get(item["content"])
                        image = Image.open(BytesIO(response.content))
                    else:
                        image = Image.open(item["content"])
                elif isinstance(item["content"], Image.Image):
                    image = item["content"]
                else:
                    raise NotImplementedError

                if image.mode != "RGB":
                    image = image.convert("RGB")

                current_message["content"].append({"type": "image", "image": image})
            elif item["type"] == "text":
                current_message["content"].append({"type": "text", "text": item["content"]})

        if current_message["content"]:
            messages.append(current_message)

        return messages

    def get_parsed_output(self, messages: List[dict]) -> str:
        """
        Get parsed output from the model
        """
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision info using qwen_vl_utils
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Process inputs
        if video_inputs:
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to(self.model.device)
        
        # Generate response
        generation_kwargs = {
            "max_new_tokens": 4096,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 0.0,
        }
        
        generated_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    
    def process_vision_info(self, messages):
        """
        Process vision information from messages using qwen_vl_utils
        """
        return process_vision_info(messages)
    
    def __del__(self):
        """
        Cleanup temporary files
        """
        for file_path in self.merged_image_files:
            try:
                import os
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
