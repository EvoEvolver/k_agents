"""Knowledge agents for lab automation."""

from mllm.config import default_models, default_options

DEFAULT_LLM_MODEL = "gpt-5.6-luna"

default_models.normal = DEFAULT_LLM_MODEL
default_models.expensive = DEFAULT_LLM_MODEL
default_models.vision = DEFAULT_LLM_MODEL

# GPT-5.6 Luna rejects MinimalLLM's default temperature of 0.2.
default_options.temperature = None
