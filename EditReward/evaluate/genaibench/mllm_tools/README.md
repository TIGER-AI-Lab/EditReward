## Usage

- Each `{model_name}_eval.py` can be run directly by `python {model_name}_eval.py` to roughly check the outputs on 3 examples. This is used for debugging purposes.

- Each `{model_name}_eval.py` defines a mllm model class which has a `__init__` method that takes in `model_id` for which checkpoint to load. This class also should have a `__call__` functions which takes in a list of messages in the following format:

- Please check `__init__.py` for a full list of supportted models.

## Example of adding a new model
### 1. Add a new model to the evaluation pipeline
- in `{model_name}_eval.py`:
```python

class NewModel():
    # support_multi_image: 'merge' images for False, and use custom image 'sequence' format for True
    support_multi_image = True
    support_video_input = False # if True, 

    def __init__(self, model_id:str="HuggingFaceM4/idefics2-8b") -> None:
        """

        Args:
            model_path (str): model name
        """
        # load models and processors
        self.model = ...
        self.processor = ...
        
    def __call__(self, inputs: List[dict]) -> str:
        """
        Generate text from images and text prompt. (batch_size=1) one text at a time.
        Args:
            inputs (List[dict]): [
                {
                    "type": "image",
                    "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg"
                },
                {
                    "type": "image",
                    "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_337180_3.jpg"
                },
                {
                    "type": "text",
                    "content": "What is difference between two images?"
                }
            ]
            Supports any form of interleaved format of image and text.
        Returns:
            str: generated text
        """
        if self.support_multi_image:
            # process images and texts ...
            generation_kwargs = {
                "max_new_tokens": 4096,
                "num_beams": 1,
                "do_sample": False,
            }
            generated_text = self.model.generate(..., generation_kwargs)
            return generated_text
        else:
            raise NotImplementedError

```

Major fields to take care of:
- If your model supports multiple images, set `support_multi_image` to `True`. Otherwise, set it to `False`. Then implement the `__call__` function accordingly.
- By default, `support_video_input` is set to `False`. And all the videos are first converted to image frames and then processed like multiple images. 
If your model supports video input, set it to `True`. Then the inputs will have `video` type inputs where the content is the video's local path. You should write your custom code to process the video input, and feed it to the model.
- The `__init__` function should load the model and processor. The model
- The `__call__` function should process the inputs and return the generated text.
- If possible, please set the `generation_kwargs` to be greedy decoding for better reproducibility. 


### 2. Register the new model in `__init__.py`
```python
MLLM_LIST = [..., "NewModel"]
...
def MLLM_Models(model_name:str):
    if ...:
        ...
    # start of your registration #
    elif model_name == "NewModel":
        from .new_model_eval import NewModel
        return NewModel
    # end of your registration #
    else:
        raise NotImplementedError
```