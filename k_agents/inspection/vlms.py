import re
from mllm import Chat
from typing import Union
import matplotlib.pyplot as plt
from PIL import Image
import io
import plotly.graph_objects as go
import os

def get_visual_analyze_prompt(func):
    """
    Get the prompt to analyze the data.

    Parameters:
        func (function): The function to get the prompt.

    Returns:
        str: The prompt to analyze the data.
    """
    return getattr(func, '_visual_prompt', None)

def run_visual_inspection(image: "Image", prompt: str, func_file_path, rescale=0.5, **kwargs) -> dict:
    """
    Ask a question about the data shape. For example, the number of peaks, or whether the data is nearly periodic.
    The answer is either True or False.
    You should use it when you need to analyze the data.

    Parameters:
        image (Image): The image to analyze.
        prompt (str): The prompt to ask.
        rescale (float): Optional. The rescale factor for the image.
        kwargs (dict): Optional. The keyword arguments to pass to the function.

    Returns:
        dict: The result of the analysis.
    """

    if not isinstance(image, Image.Image):
        assert isinstance(image, plt.Figure) or isinstance(image,
                                                           go.Figure), "The image must be a PIL image or a Matplotlib or Plotly figure."
        image = matplotlib_plotly_to_pil(image)

    chat = prepare_visual_inspection_chat(prompt, func_file_path)
    chat.system_message = ("You are a helpful visual assistant that able to provide analysis on images or plots. "
                       "Please return your message in a json format with keys analysis(str, single paragraph) "
                       "and 'success'(boolean)")
    chat.add_user_message("Here is the input image:")
    chat.add_image_message(image)
    res = chat.complete(parse="dict", cache=True, **kwargs)

    return res

def prepare_visual_inspection_chat(raw_prompt, func_file_path):
    # extract all the Image("...") or Image('...') patterns and break the raw_prompt into parts
    # for example "123 Image("dasda") 123" should be broken into ['123 ', 'Image("dasda")', ' 123']
    func_dir = os.path.dirname(func_file_path)
    image_pattern = re.compile(r'Image\(["\'].*?["\']\)')
    image_matches = image_pattern.findall(raw_prompt)
    parts = image_pattern.split(raw_prompt)
    path_pattern = re.compile(r'Image\(["\'](.*?)["\']\)')

    merged_parts = []
    for part, image_match in zip(parts, image_matches):
        merged_parts.append(part)
        #
        path = path_pattern.search(image_match).group(1)
        merged_parts.append({"src": os.path.join(func_dir, path), "type": "image"})
    if len(parts) > len(image_matches):
        merged_parts.append(parts[-1])

    chat = Chat()
    for part in merged_parts:
        if isinstance(part, str):
            trimmed = part.strip()
            if len(trimmed) > 0:
                chat.add_user_message(trimmed)
        else:
            chat.add_image_message(part["src"])

    return chat


def matplotlib_plotly_to_pil(fig: Union[go.Figure, plt.Figure]):
    """
    Convert a Matplotlib or Plotly figure to a PIL image.

    Parameters:
        fig (Union[go.Figure, plt.Figure]): The Matplotlib or Plotly figure.

    Returns:
        Image: The PIL image.
    """

    # Save the Matplotlib figure to a BytesIO object

    buf = io.BytesIO()
    if isinstance(fig, go.Figure):
        if os.name == 'nt':
            engine = "orca"
        else:
            engine = "kaleido"
        fig.write_image(buf,format='png',engine=engine)
    elif isinstance(fig, plt.Figure):
        fig.savefig(buf, format='png')
    else:
        raise ValueError(f"The input must be a Matplotlib or Plotly figure. Got {type(fig)}.")

    buf.seek(0)

    # Open the image with PIL
    image = Image.open(buf)
    return image