import re

from k_agents.agent_group.w_memory import WMemoryItem

"""
# Working memory items for code
"""


class CodeWMemoryItem(WMemoryItem):
    def __init__(self, code: str, tag="code"):
        super().__init__(code, tag=tag)

    def get_stimuli(self):
        do_stimuli = extract_func_parameters("do", self.content)
        comment_stimuli = extract_code_comment(self.content)
        stimuli = do_stimuli + comment_stimuli
        return stimuli


def extract_func_parameters(func_name, code):
    pattern = re.compile(rf'{func_name}\(\"((.|\n)*?)\"\s*\)', re.DOTALL)
    matches = pattern.findall(code)
    pattern2 = re.compile(rf'{func_name}\(\'((.|\n)*?)\'\s*\)', re.DOTALL)
    matches.extend(pattern2.findall(code))
    res = [x[0] for x in matches]
    res = [x.strip() for x in res]
    res = [x.strip("\"") for x in res]
    return res


def extract_code_comment(code) -> list[str]:
    """
    Extract all the comment in the form of # comment
    """
    pattern = re.compile(r"#\s*(.*)")
    comments = pattern.findall(code)
    docstring_pattern = re.compile(r'\"\"\"((.|\n)*?)\"\"\"', re.DOTALL)
    docstring_pattern2 = re.compile(r"\'\'\'((.|\n)*?)\'\'\'", re.DOTALL)
    docstrings = docstring_pattern.findall(code)
    docstrings.extend(docstring_pattern2.findall(code))
    docstrings = [x[0] for x in docstrings]
    return comments + docstrings
