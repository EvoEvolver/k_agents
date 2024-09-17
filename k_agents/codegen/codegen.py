from mllm import Chat
from mllm.utils.parser import Parse

from k_agents.ideanet.code_wmemory import CodeWMemoryItem
from k_agents.ideanet.lt_memory import Idea, IdeaResult, LongTermMemory, RecallResult
from k_agents.ideanet.w_memory import WorkingMemory, WMemoryItem
from k_agents.variable_table import VariableTable


class CodegenIdea(Idea):
    """
    Generate the code based on the working memory
    Will put the generated code in the working memory
    """

    def __init__(self):
        super().__init__("CodegenIdea")

    def get_score(self, w_memory: WorkingMemory):
        if not w_memory.has_tag("code_suggestion"):
            return -1.0
        return 1.0

    def run_idea(self, w_memory: WorkingMemory) -> IdeaResult:
        instruction = w_memory.extract_tag_contents("instruction")[0]
        available_variables = w_memory.extract_tag_contents("available_variables")
        if len(available_variables) == 0:
            available_variables = "There is no available variables"
        else:
            available_variables = available_variables[0]
        code_suggestions = w_memory.extract_tag_contents("code_suggestion")
        code_suggestions = ["<code_suggestion>\n" + suggestion + "\n</code_suggestion>\n" for suggestion in
                            code_suggestions]
        code_suggestions = "".join(code_suggestions)
        notices = """
        - Call exactly one time to the experiment function / class in this edit.
        - Every class or function call will include the data analysis inside the call automatically so there is no need to do data analysis separately.
        - Always use named arguments to call functions or classes.
        - Store the return value of the call functions or classes to a variable.
        """
        prompt = f"""
Your task is to generate new code for the context described below.        
<context>
<available_variables>
{available_variables}
</available_variables>
<code_to_complete> 
# [slot: {instruction}]
</code_to_complete>
You must use the following code_suggestions to generate the code:
{code_suggestions}
</context>
<requirements>
You are required to adopt code from one of <code_suggestion> that can be used to fill the slot in <code_to_complete>.
The adopted code should absolutely just be what should appear in the place of # [slot]. 
Some of the <code_suggestion> might be misleading. But you must pick the most relevant one.
You should first output an analysis of which code suggestion should be used to fill the slot in <code_to_complete>.
Then, wrapped by ```python and ```, output the new code that can fill the slot in <code_to_complete>. The last line of the generated code must be in the format: `experiment_<name> = <ExperimentName>(argument1,argument2, ...)`. The code must be executable. 
</requirements>
        """
        chat = Chat(prompt)
        code = chat.complete(parse="quotes", expensive=True)
        idea_res = IdeaResult(self, True)

        if code.startswith("```"):
            code = Parse.quotes(code)

        code_item = CodeWMemoryItem(code, tag="attempted_code")
        idea_res.add_new_wm_item(code_item)
        idea_res.tags_to_remove = ["attempted_code", "code_suggestion"]  # remove the old attempted code
        return idea_res


class CodegenModel:

    def __init__(self):
        self.lt_memory = LongTermMemory()
        self.codegen_idea = CodegenIdea()
        self.n_recall_items = 10
        self._cached_recall_res = None

    def recall(self, wm: WorkingMemory, n_recall_items=None) -> RecallResult:
        """
        Recall ideas from long term memory, using what is currently in the working memory.

        :param wm: the working memory to stimuli ideas
        :return: the result of triggered ideas
        """
        if n_recall_items is None:
            n_recall_items = self.n_recall_items
        res = self.lt_memory.recall_by_wm(wm, top_k=n_recall_items)
        self._cached_recall_res = res
        return res

    def codegen(self, wm: WorkingMemory, recall_res: dict = None) -> str:
        """
        Generate code from working memory, updates working memory with recalled ideas in the process.

        Parameters:
        - wm: the working memory to generate code from
        - recall_res: the recall result from the long term memory

        Preconditions:
        - there exists an item in wm tagged with 'completed_code' after at most 100 recalls.
        """
        if recall_res is None:
            if self._cached_recall_res is None:
                recall_res = self.recall(wm)
            else:
                recall_res = self._cached_recall_res
        n_recall_items = self.n_recall_items
        for i in range(3):
            wm.update_by_recall_res(recall_res, to_tick=True)
            if len(wm.extract_tag_contents("code_suggestion"))>0:
                break
            else:
                print("No code suggestion found. Recall more ideas.")
                n_recall_items += 2
                recall_res = self.recall(wm, n_recall_items)

        idea_res = self.codegen_idea.run_idea(wm)
        recall_res = RecallResult([idea_res])
        wm.update_by_recall_res(recall_res, to_tick=False)
        code = wm.extract_tag_contents("attempted_code")
        if len(code) > 0:
            return code[0]


def get_code_from_wm(wm: WorkingMemory) -> str:
    """
    Extracts code from the working memory that contains code items.

    Parameters:
        wm (WorkingMemory): The working memory to extract code from.

    Returns:
        str: The extracted code.
    """
    code = ""
    for item in wm._items:
        if isinstance(item, CodeWMemoryItem):
            code = item.content
            break
    return code


def get_codegen_wm(description: str, var_table: VariableTable) -> WorkingMemory:
    """
    Prepares working memory for code generation based on a description and variable table.

    Parameters:
        description (str): The description of the code to generate.
        var_table (VariableTable): The variable table to use in the code generation.
        hint (str): The hint to display in the working memory.
    Returns:
        WorkingMemory: The working memory prepared for code generation.
    """

    wm = WorkingMemory()
    if not var_table.is_empty():
        var_table_in_prompt = var_table.get_local_prompt()
        var_table_item = WMemoryItem(var_table_in_prompt, "available_variables")
        var_table_item.set_no_prompt()
        var_table_item.attrs["_table_obj"] = var_table
        wm.add_item(var_table_item.set_no_stimuli())
    wm.add_item(WMemoryItem(description, tag="instruction"))
    return wm