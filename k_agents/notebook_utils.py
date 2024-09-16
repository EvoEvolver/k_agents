import uuid

from IPython.core.display import HTML, Javascript
from IPython.core.display_functions import display


def show_spinner(text="Processing..."):
    """
    Displays a spinner with a specified text to indicate processing.

    Args:
    text (str): The text to display alongside the spinner.

    Returns:
    str: A unique identifier for the spinner element.
    """
    # Generate a unique identifier for the spinner
    spinner_id = str(uuid.uuid4())

    # Create and display the spinner HTML element
    spinner_html = HTML(f"""
    <div id="{spinner_id}" style="font-size:16px;">
        <i class="fa fa-spinner fa-spin"></i> {text}
    </div>
    """)
    display(spinner_html)

    # Return the unique ID for future reference (e.g., to hide the spinner)
    return spinner_id


def hide_spinner(spinner_id):
    """
    Hides and removes the spinner identified by spinner_id from the display.

    Args:
    spinner_id (str): The unique identifier of the spinner to hide.
    """
    # Create and execute JavaScript to hide and remove the spinner element
    hide_spinner_js = Javascript(f"""
    var spinnerElement = document.getElementById('{spinner_id}');
    if (spinnerElement) {{
        spinnerElement.remove();  // Remove the spinner element from the DOM
    }} else {{
        // Retry removing the spinner after a short delay if not immediately found
        window.setTimeout(() => {{
            var spinnerElement = document.getElementById('{spinner_id}');
            if (spinnerElement) {{
                spinnerElement.remove();
            }}
        }}, 100);
    }}
    """)
    display(hide_spinner_js)


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
