"""
Install the Google AI Python SDK

$ pip install google-generativeai

See the getting started guide for more information:
https://ai.google.dev/gemini-api/docs/get-started/python
"""

import requests
import time
import pathlib
from PIL import Image
from io import BytesIO
import os
from typing import List
from urllib.parse import urlparse
import google.generativeai as genai
import tempfile

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def upload_to_gemini(input, mime_type=None):
    """Uploads the given file or PIL image to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    if isinstance(input, str):
        # Input is a file path
        file = genai.upload_file(input, mime_type=mime_type)
    elif isinstance(input, Image.Image):
        # Input is a PIL image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            input.save(tmp_file, format="JPEG")
            tmp_file_path = tmp_file.name
        file = genai.upload_file(tmp_file_path, mime_type=mime_type or "image/jpeg")
        os.remove(tmp_file_path)
    else:
        raise ValueError("Unsupported input type. Must be a file path or PIL Image.")

    #print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def save_image_from_url(url, base_save_directory='tmp', file_name=None):
    # Parse the URL to create a directory path
    parsed_url = urlparse(url)
    url_path = os.path.join(parsed_url.netloc, parsed_url.path.lstrip('/'))
    save_directory = os.path.join(base_save_directory, os.path.dirname(url_path))
    
    # Create the directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Get the image from the URL
    response = requests.get(url)
    if response.status_code == 200:
        # Open the image
        image = Image.open(BytesIO(response.content))
        
        # Set the file name if not provided
        if not file_name:
            file_name = os.path.basename(parsed_url.path)
        
        # Save the image locally
        file_path = os.path.join(save_directory, file_name)
        image.save(file_path)
        
        return file_path
    else:
        raise Exception(f"Failed to retrieve image from URL. Status code: {response.status_code}")

def save_image_to_tmp(image):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        image.save(tmp_file, format="JPEG")
        tmp_file_path = tmp_file.name
    return tmp_file_path
class Gemini():
    support_multi_image = True
    support_video_input = True
    def __init__(self, model_name="gemini-1.5-pro-latest"):
        # Create the model
        # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
        generation_config = {
          "temperature": 0,
          "top_p": 1.0,
          "max_output_tokens": 16384,
          "response_mime_type": "text/plain",
        }
        safety_settings = [
          {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
          },
          {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
          },
          {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
          },
          {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
          },
        ]
        self.model = genai.GenerativeModel(
          model_name=model_name,
          safety_settings=safety_settings,
          generation_config=generation_config,
        )

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
        has_image = any(x["type"] == "image" for x in inputs)
        has_video = any(x["type"] == "video" for x in inputs)
        text_prompt = "\n".join([x["content"] for x in inputs if x["type"] == "text"])

        contents = []
        for item in inputs:
            if item["type"] == "image":
              if isinstance(item["content"], str):
                if item["content"].startswith("http"):
                  image_file = save_image_from_url(item["content"])
              elif isinstance(item["content"], Image.Image):
                image_file = save_image_to_tmp(item["content"])
              else:
                raise ValueError("Unsupported input type. Must be a file path or URL.")
              image_file = genai.upload_file(path=image_file)
              image_file = genai.get_file(name=image_file.name)
              contents.append(image_file)
            elif item["type"] == "video":
              if isinstance(item["content"], str) or isinstance(item["content"], pathlib.Path):
                video_file = genai.upload_file(path=item["content"])
                                # Check whether the file is ready to be used.
                while video_file.state.name == "PROCESSING":
                    print('.', end='')
                    time.sleep(3)
                    video_file = genai.get_file(video_file.name)

                if video_file.state.name == "FAILED":
                  raise ValueError(video_file.state.name)
                contents.append(video_file)
              else:
                raise ValueError("Unsupported input type. Must be a file path. but get {}".format(type(item["content"])))
            elif item["type"] == "text":
              contents.append(item["content"])
        response = self.model.generate_content(contents, request_options={"timeout": 600})
        try:
          return response.text
        except Exception as e:
          print(e)
          print(response.prompt_feedback)
          return str(response.prompt_feedback)
if __name__ == "__main__":
    model = Gemini()
    # difference
    difference_exs = [
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
        },
    ]
    # print("### 0 shot")
    # print(model(zero_shot_exs))
    # print("### 1 shot")
    # print(model(one_shot_exs))
    # print("### 2 shot")
    # print(model(two_shot_exs))
    print("### difference")
    print(model(difference_exs))