"""pip install transformers>=4.35.2 transformers_stream_generator torchvision tiktoken chardet matplotlib
""" 
import tempfile
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from typing import List

class MiMoVLUltraFast():
    support_multi_image = True
    support_video_input = False
    merged_image_files = []
    
    def __init__(self, model_id: str = "/mnt/f/NewGit/GenAI-Bench-main/models/MiMo-VL-7B-SFT-2508") -> None:
        """
        Args:
            model_id (str): MiMo-VL model path, default to local MiMo-VL-7B-SFT-2508
        """
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, 
            device_map="auto", 
            trust_remote_code=True, 
            torch_dtype="auto"
        ).eval()
    
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
            # Process multiple images and text
            messages = self.prepare_messages(inputs)
            response = self.get_parsed_output(messages)
            return response
        else:
            raise NotImplementedError
    
    def prepare_messages(self, inputs: List[dict]) -> List[dict]:
        """
        Prepare messages in the format expected by MiMo-VL
        """
        messages = []
        current_message = {"role": "user", "content": []}
        
        for item in inputs:
            if item["type"] == "image":
                # Handle image content
                if isinstance(item["content"], str):
                    if item["content"].startswith("http"):
                        # Download image from URL
                        import requests
                        from io import BytesIO
                        response = requests.get(item["content"])
                        image = Image.open(BytesIO(response.content))
                    else:
                        # Local file path
                        image = Image.open(item["content"])
                elif isinstance(item["content"], Image.Image):
                    image = item["content"]
                else:
                    raise NotImplementedError
                
                # Convert to RGB if necessary
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
        Generate response using MiMo-VL model
        """
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision info
        image_inputs, video_inputs = self.process_vision_info(messages)
        
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
        Process vision information from messages
        """
        image_inputs = []
        video_inputs = []
        
        for message in messages:
            if "content" in message:
                for content in message["content"]:
                    if content["type"] == "image":
                        image_inputs.append(content["image"])
                    elif content["type"] == "video":
                        video_inputs.append(content["video"])
        
        return image_inputs, video_inputs
    
    def __del__(self):
        for image_file in self.merged_image_files:
            if os.path.exists(image_file):
                os.remove(image_file)

if __name__ == "__main__":
    model = MiMoVLUltraFast()
    
    # Test with multiple images
    test_inputs = [
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
            "content": "What is the difference between these two images?"
        }
    ]
    
    print("### Testing MiMo-VL-7B-SFT-2508")
    print(model(test_inputs))
