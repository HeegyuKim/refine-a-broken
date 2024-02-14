import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():

    config = default_config()

    config.tokenizer_paths = [
        "berkeley-nest/Starling-LM-7B-alpha",
    ]
    config.tokenizer_kwargs = [{"use_fast": False}]
    config.model_paths = [
        "berkeley-nest/Starling-LM-7B-alpha",
    ]
    config.conversation_templates = ["vicuna"]

    return config
