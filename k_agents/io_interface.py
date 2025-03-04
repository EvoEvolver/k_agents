import uuid

import mllm.utils.maps
from IPython.core.display import Javascript
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from IPython.display import display, HTML

display_impl = print

def set_jupyter_display_impl():
    global display_impl
    display_impl = display

# check if in the Jupyter notebook
try:
    from IPython import get_ipython
    if get_ipython():
        set_jupyter_display_impl()
except:
    pass

def set_streamlit_display_impl():
    global display_impl
    from streamlit import html
    from stqdm import stqdm
    mllm.utils.maps.default_parallel_map_config["pbar"] = stqdm
    display_impl = html


def show_spinner(text="Processing..."):
    """
    Displays a spinner with a specified text to indicate processing.

    Args:
    text (str): The text to display alongside the spinner.

    Returns:
    str: A unique identifier for the spinner element.
    """
    ...

def hide_spinner(spinner_id):
    """
    Hides and removes the spinner identified by spinner_id from the display.

    Args:
    spinner_id (str): The unique identifier of the spinner to hide.
    """
    ...


class Spinner:
    def __init__(self, text="Processing..."):
        self.text = text
        self.spinner_id = None

    def show(self):
        self.spinner_id = show_spinner(self.text)

    def hide(self):
        if self.spinner_id:
            hide_spinner(self.spinner_id)
            self.spinner_id = None

    def __enter__(self):
        self.show()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.hide()


def change_spinner_text(spinner_id, new_text):
    """
    Changes the text of the spinner identified by spinner_id.

    Args:
    spinner_id (str): The unique identifier of the spinner.
    new_text (str): The new text to display in the spinner.
    """
    # Create and execute JavaScript to change the text of the spinner element
    change_text_js = Javascript(f"""
    var spinnerElement = document.getElementById('{spinner_id}');
    if (spinnerElement) {{
        spinnerElement.textContent = '{new_text}';  // Change the text content of the spinner element
    }} else {{
        // Retry changing text after a short delay if the spinner is not immediately found
        window.setTimeout(() => {{
            var spinnerElement = document.getElementById('{spinner_id}');
            if (spinnerElement) {{
                spinnerElement.textContent = '{new_text}';
            }}
        }}, 50);
    }}
    """)
    display(change_text_js)


def dict_to_html(data_dict) -> str:
    """
    Converts a dictionary into an HTML string with keys in bold and values formatted with repr(),
    styled to fit within a chat bubble.

    Args:
    data_dict (dict): The dictionary to convert to HTML.

    Returns:
    str: The HTML formatted string.
    """
    html_parts = []
    for key, value in data_dict.items():
        formatted_value = repr(value)
        html_parts.append(f"<strong>{key}</strong>: {formatted_value}<br>")
    return ''.join(html_parts)


def code_to_html(code: str):
    """
    Convert Python code to HTML using Pygments.

    Args:
    code (str): Python code to convert.

    Returns:
    str: HTML formatted string representing the Python code.
    """
    formatter = HtmlFormatter()
    formatted_code = highlight(code, PythonLexer(), formatter)
    return f"""
        <div style="background: #f8f8f8; padding: 10px; border-radius: 5px; overflow: auto; margin-top: 8px;">
            {formatted_code}
        </div>
"""


def display_chat(agent_name: str, content: str, background_color: str = "light_blue"):
    """
    Display a chat message formatted with the given parameters.

    Args:
    agent_name (str): Name of the agent speaking.
    background_color (str): Background color for the chat bubble.
    content (str): Content of the message, which may include plain text or HTML.

    Returns:
    str: HTML formatted string representing the chat message.
    """

    background_color_set = {
        'light_orange': '#FFF7EB',
        'light_blue': '#F0F8FF',
        'light_green': '#F0FFF0',
        'light_red': '#FFF0F5',
        'light_yellow': '#FFFFE0',
        'light_purple': '#F8F8FF',
        'light_pink': '#FFF0F5',
        'light_cyan': '#E0FFFF',
        'light_lime': '#F0FFF0',
        'light_teal': '#E0FFFF',
        'light_mint': '#F0FFF0',
        'light_lavender': '#F8F8FF',
        'light_peach': '#FFEFD5',
        'light_rose': '#FFF0F5',
        'light_amber': '#FFFFE0',
        'light_emerald': '#F0FFF0',
        'light_platinum': '#F1EEE9',
    }

    if background_color in background_color_set:
        background_color = background_color_set[background_color]

    html = f'''
    <p style="background-color: {background_color}; padding: 20px; border-radius: 8px; color: #333;">
        <strong>{agent_name}:</strong> {content}
    </p>
    '''
    display_impl(HTML(html))
