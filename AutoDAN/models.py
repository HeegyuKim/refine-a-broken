
def get_developer(model_name):
    developer_dict = {"llama2": "Meta", "vicuna": "LMSYS",
                      "guanaco": "TheBlokeAI", "WizardLM": "WizardLM",
                      "tinyllama": "tinyllama",
                      "tinymistral": "tinymistral",
                      "Starling-LM-7B-alpha": "Starling-LM-7B-alpha",
                      "zephyr": "HuggingFaceH4",
                      "starling-7b": "declare-lab",
                      "mpt-chat": "MosaicML", "mpt-instruct": "MosaicML", "falcon": "TII"}
    return developer_dict[model_name]

model_path_dicts = {"llama2": "meta-llama/Llama-2-7b-chat-hf", "vicuna": "./models/vicuna/vicuna-7b-v1.3",
                    "guanaco": "./models/guanaco/guanaco-7B-HF", "WizardLM": "./models/WizardLM/WizardLM-7B-V1.0",
                    "mpt-chat": "./models/mpt/mpt-7b-chat", "mpt-instruct": "./models/mpt/mpt-7b-instruct",
                    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "tinymistral": "Felladrin/TinyMistral-248M-SFT-v4",
                    "starling-7b": "declare-lab/starling-7B",
                    "Starling-LM-7B-alpha": "berkeley-nest/Starling-LM-7B-alpha",
                    "zephyr": "HuggingFaceH4/zephyr-7b-beta",
                    "falcon": "./models/falcon/falcon-7b-instruct"}