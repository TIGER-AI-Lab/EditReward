import os
from typing import List, Union, Optional, Dict
from transformers.image_utils import load_image
from .model_utils.openai_utils import local_image_to_data_url
import openai

class Qwen2_5_7B_Instruct():
    support_multi_image = True
    merged_image_files = []
    
    def __init__(self, model_path: str = "qwen2.5-7b-instruct", base_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
        """
        Initialize Qwen2.5-7B-Instruct API client
        
        Args:
            model_path (str): Model name for API calls
            base_url (str): API base URL
            api_key (str): API key for authentication
        """
        self.model_path = model_path
        self.base_url = base_url or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def __call__(self, inputs: List[dict]) -> str:
        """
        Process inputs and return model response
        
        Args:
            inputs (List[dict]): List of input messages with type and content
        Returns:
            str: Model response
        """
        if self.support_multi_image:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an AI assistant that helps people find information."
                        }
                    ]
                }
            ]
            messages += [
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            for message in inputs:
                if message["type"] == "image":
                    # Convert image to proper data URL format for API
                    image_data_url = local_image_to_data_url(message["content"])
                    messages[-1]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    )
                elif message["type"] == "text":
                    messages[-1]["content"].append(
                        {
                            "type": "text",
                            "text": message["content"]
                        }
                    )
                else:
                    raise NotImplementedError

            # Direct API call to avoid message format conversion issues
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            response = client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=4000
            )
            
            return response.choices[0].message.content
        else:
            raise NotImplementedError
