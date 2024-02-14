
import os
os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():

    config = default_config()

    config.transfer = True
    config.logfile = ""

    config.progressive_goals = False
    config.stop_on_success = False
    config.tokenizer_paths = [
        "declare-lab/starling-7B"
    ]
    config.tokenizer_kwargs = [{"use_fast": False}, {"use_fast": False}]
    config.model_paths = [
        "declare-lab/starling-7B"
    ]
    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": False},
        {"low_cpu_mem_usage": True, "use_cache": False}
    ]
    config.conversation_templates = ["vicuna", "vicuna"]
    config.devices = ["cuda:0"]

    return config