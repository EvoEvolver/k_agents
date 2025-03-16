import os.path

import markdown
from bs4 import BeautifulSoup
from markdownify import markdownify
from mllm import Chat
from mllm.utils.maps import p_map

from k_agents.agent_group.agent_group import AgentGroup, EmbedAgent, AgentResult
from k_agents.agent_group.w_memory import WorkingMemory
from k_agents.variable_table import VariableTable


def get_experiment_name_for_procedure(title):
    # filter all the non-alphanumeric characters
    title = "".join([c for c in title if c.isalnum() or c == " "])
    # split the title by space
    title_segments = title.split(" ")
    new_title_segments = []
    for seg in title_segments:
        # set the first character to uppercase
        new_title_segments.append(seg[0].upper() + seg[1:])
    # join the segments
    title = "".join(new_title_segments)
    return title


class ProcedureTranslationAgent(EmbedAgent):
    _n_procedure_agents = 1

    def __init__(self, title: str, steps: str, background: str, embed_src: list[str]):
        self.title = title
        self.steps = steps
        self.background = background
        self.label = f"Experiment_{get_experiment_name_for_procedure(title)}"
        ProcedureTranslationAgent._n_procedure_agents += 1
        super().__init__(f"ProcedureTranslationAgent for {title}", embed_src)

    def run_agent(self, w_memory: WorkingMemory) -> AgentResult:
        instruction = w_memory.extract_tag_contents("instruction")[0]
        available_variables = w_memory.extract_tag_contents("available_variables")
        if len(available_variables) == 0:
            available_variables = "There is no available variables"
        else:
            available_variables = available_variables[0]
        prompt = f"""
You are trying to rewrite the following instruction based on your knowledge to suit more general parameters.
<input_instruction>
{instruction}
</input_instruction>
The following variables names are defined in the context and you can directly use the variables names.
<available_variables>
{available_variables}
</available_variables>
By your knowledge, there is an existing instruction that is available
<knowledge>
<instruction>
{self.title}
</instruction>
</knowledge>
<requirements>
You are required to output a JSON dict with the following keys
- "parameter_specification" (string): The specification of the parameters of the input_instruction 
- "analysis" (string): An analysis of the relation between the input_instruction and your knowledge. You should notice that your knowledge is likely to be improper if the experiment name contains different keywords. However, the parameter_specification of the experiment can be different.
- "proper" (bool): Whether the input_instruction can be regarded as a proper instance of the experiment in your knowledge.
- "rewritten_instruction" (string): The input_instruction rewritten in a way based on the instruction in your knowledge. You should not change the parameter specification of the input_instruction.
- "parameter_mapping" (dict): A mapping from the parameters in the input_instruction to the parameters in the rewritten_instruction. The keys should be the parameters in the rewritten_instruction and the values should be the parameters in the input_instruction or the value from the description.
- "annotation" (string): A concise annotation that describe how you are confident that the rewritten_instruction is correct.
</requirements>
"""
        chat = Chat(prompt)
        res = chat.complete(parse="dict", expensive=True)
        if not res["proper"]:
            return AgentResult(self, False)
        arg_in_code = []
        for key, value in res["parameter_mapping"].items():
            arg_in_code.append(f", {key}={value}")
        arg_in_code = "".join(arg_in_code)
        if "annotation" in res and res["annotation"] is not None:
            annotation_in_prompt = res["annotation"].replace("\n", "\n# ")
        else:
            annotation_in_prompt = ""
        code_suggestion = f'''
experiment_instance = {self.label}(instruction="""{res["rewritten_instruction"]}""" {arg_in_code})  
# {annotation_in_prompt}      
'''
        agent_res = AgentResult(self, True)
        agent_res.add_new_wm_content(code_suggestion, tag="code_suggestion")
        return agent_res

    def get_score(self, w_memory: WorkingMemory):
        stimuli = w_memory.stimuli
        for stim in stimuli:
            if self.title in stim:
                return 2.0
        return EmbedAgent.get_score(self, w_memory)


def extract_procedure_contents(markdown_path):
    with open(markdown_path, "r") as f:
        src = f.read()
    # Get html of the markdown
    html = markdown.markdown(src)
    # Parse the html
    soup = BeautifulSoup(html, "html.parser")
    title_html_list = []
    # Find the contents between <h1>
    for h1 in soup.find_all("h1"):
        siblings = []
        title = h1.text
        # Get the following siblings of h1
        for sibling in h1.next_siblings:
            # If the sibling is a tag, break the loop
            if sibling.name == "h1":
                break
            siblings.append(sibling)
        sibling_html = "".join([str(sibling) for sibling in siblings])
        title_html_list.append((title, sibling_html))
        # Convert the html to markdown with sections start with #
        # sibling_md = markdownify(sibling_html, heading_style="ATX")
        # procedures.append((title, sibling_md))
    procedure_list = []
    for title, sibling_html in title_html_list:
        # extract header with title Steps
        steps = ""
        background = ""
        result = None
        for h2 in BeautifulSoup(sibling_html, "html.parser").find_all("h2"):
            if h2.text == "Steps":
                steps = extract_text_after_section(h2)
            elif h2.text == "Background":
                background = extract_text_after_section(h2)
            elif h2.text == "Result":
                result = extract_text_after_section(h2)
        procedure_list.append({
            "title": title,
            "background": background,
            "steps": steps,
            "result": result
        })
    return procedure_list


def extract_text_after_section(section_tag):
    siblings = []
    # get all siblings until next h2
    for sibling in section_tag.next_siblings:
        if sibling.name == "h2":
            break
        siblings.append(sibling)
    text = "".join([str(sibling) for sibling in siblings])
    text = markdownify(text, heading_style="ATX").strip()
    return text

def imagine_applications_for_doc(title, background):
    prompt = f"""
You are trying to produce imperative sentences that would invoke the execution a certain experiment based on its title and background description.
<title>
{title}
</title>
<background>
{background}
</background>
<example>
Here are a few of examples of imperative sentences:
- Run the <Experiment Name> with duts=duts and start=0.0
- Please execute the <Experiment Name> experiment with end=2.0
- Do the <Experiment Name> drag experiment.
</example>
<instruction>
You should output a JSON dict. The keys should be string of indices of the sentences and the values should be the sentences. 
Each sentence should be complete and independent. Name of the experiment should be transformed to natural language and be mentioned.
The sentences should be imperative and should be based on the documentation.
You should output 4 sentences.
</instruction>"""
    chat = Chat(prompt)
    res = chat.complete(parse="dict", expensive=True, cache=True)
    values = list(res.values())
    return values


def generate_agent_from_procedure(procedure):
    title = procedure["title"]
    background = procedure["background"]
    steps = procedure["steps"]
    embed_src = imagine_applications_for_doc(title, background)
    agent = ProcedureTranslationAgent(title, steps, background, embed_src + [title])
    return agent


def get_exp_class(procedure):
    from k_agents.execution.agent import AutomatedExperiment
    class AutomatedExperiment_x(AutomatedExperiment):
        _title = procedure["title"]
        _original_steps = procedure["steps"]
        _background = procedure["background"]
        _expected_result = procedure["result"]

    def exp_runner(instruction=None, **kwargs):
        assert instruction is not None, "instruction is required"
        exp = AutomatedExperiment_x(instruction, kwargs)
        return exp

    return exp_runner


def extract_procedures_to_agent_group(markdown_paths: list[str], agent_group, var_table):
    all_procedures = []
    for markdown_path in markdown_paths:
        procedures = extract_procedure_contents(markdown_path)
        all_procedures.extend(procedures)
    for procedure, agent in p_map(generate_agent_from_procedure, all_procedures):
        var_table.add_variable(agent.label, get_exp_class(procedure))
        agent_group.add_agent(agent)