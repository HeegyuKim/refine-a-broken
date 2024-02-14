

import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():

    config = default_config()

    config.tokenizer_paths = [
        "declare-lab/starling-7B",
    ]
    config.tokenizer_kwargs = [{"use_fast": False}]
    config.model_paths = [
        "declare-lab/starling-7B",
    ]
    config.conversation_templates = ["vicuna"]

    return config
