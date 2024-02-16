import fire
from typing import Optional, List
import jsonlines
import os
from tqdm import tqdm
from .gen_utils import Generator

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    AutoConfig,
    StoppingCriteria,
)
from .prompt_templates import PROMPT_TEMPLATES
from .question_datasets import get_dataset
from .jailbreak_utils import JAILBREAK_FUNCTIONS
from .gen_utils import generators
from .rm_utils import RewardAPI


from copy import deepcopy

import torch

def build_model_types(dtype: str, device: str, num_gpus: int):
    if num_gpus > 1:
        return {"torch_dtype": DTYPES[dtype]}
    elif dtype == "int8":
        return {"load_in_8bit": True, "device_map": "auto"}
    elif dtype == "int4":
        return {"load_in_4bit": True, "device_map": "auto"}
    else:
        return {"torch_dtype": DTYPES[dtype], "device_map": device}

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, eos_list: List[str]):
        self.eos_list = [tokenizer.encode(x) for x in eos_list]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for eos_sequence in self.eos_list:
            last_ids = input_ids[:,-len(eos_sequence):].tolist()
            if eos_sequence in last_ids:
                return True
        return False

@torch.no_grad()
def main(
    model_id: str,
    testset: str,
    output_filename: Optional[str] = None,
    device: str = "auto",
    generator_name: str = "default",
    batch_size: int = 1,
    datasize: Optional[int] = None,
    dtype: str = "float16",
    reward_api: Optional[str] = None,
    limit: Optional[int] = -1,
    prompt_template: Optional[str] = None,
    system_prompt: Optional[str] = None,
    model_revision: Optional[str] = None,
    tokenizer_id: Optional[str] = None,
    peft_model_id: Optional[str] = None,
    peft_model_revision: Optional[str] = None,
    trust_remote_code: bool = False,
    jailbreak: Optional[str] = None,
    score_response_only: bool = False,
    instruction_suffix: Optional[str] = None,
    response_prefix: Optional[str] = None,
    mesh: Optional[str] = "fsdp",
    do_sample=True,
    num_gpus=1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    max_length: Optional[int] = 1024,
    max_new_tokens: Optional[int] = 512,
    additional_eos: Optional[str] = None,
    print_generation: bool = True
):
    if reward_api:
        reward_server = RewardAPI(reward_api, response_only=score_response_only)
    if output_filename is None:
        output_filename = f"data/{model_id}/{testset}.json"
    
    dataset = get_dataset(testset, model_id=model_id)
    print(dataset)
    if datasize:
        dataset = dataset.shuffle(seed=42).select(range(datasize))

    if jailbreak:
        new_dataset = []
        if jailbreak == "*":
            jailbreak = list(JAILBREAK_FUNCTIONS.keys())
        else:
            jailbreak = jailbreak.split(",")

        print(jailbreak)
        for jb in jailbreak:
            print(jb)
            jailbreak_func = JAILBREAK_FUNCTIONS[jb]

            for i, item in enumerate(dataset):
                # if limit and i >= limit:
                #     break

                item["jailbreak_func"] = jb
                item["conversations"][-1]['content'] = jailbreak_func(item["conversations"][-1]['content'])
                new_dataset.append(item)

        limit = limit * len(jailbreak)
        dataset = new_dataset
        print(len(dataset))
    if os.path.exists(output_filename):
        with jsonlines.open(output_filename) as fin:
            skip_lines = len(list(fin))
            print(f"{skip_lines} items are already generated. Skip them.")
        
        if skip_lines >= len(dataset):
            print("skip this input, already handled", output_filename)
            return
    else:
        skip_lines = 0

    if tokenizer_id is None or len(tokenizer_id.strip()) == 0:
        tokenizer_id = model_id

    if generator_name.startswith("api/"):
        model = model_id
        tokenizer = model_id
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            revision=model_revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        trust_remote_code=trust_remote_code,
        # low_cpu_mem_usage=False, 
        offload_state_dict=True,
        **(build_model_types(dtype, device, num_gpus)),
        ).eval()

    if prompt_template:
        tokenizer.chat_template = PROMPT_TEMPLATES[prompt_template]
    elif hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        pass
    else:
        tokenizer.chat_template = PROMPT_TEMPLATES[model_id]

    gen_args = dict(
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        use_cache=True,
    )
    gen_args_save = gen_args.copy()
    if additional_eos:
        gen_args_save.pop("stopping_criteria")
    if additional_eos:
        gen_args["stopping_criteria"] = [EosListStoppingCriteria(tokenizer, [additional_eos])]

    model_args = dict(
        model_id=model_id,
        model_revision=model_revision,
        peft_model_id=peft_model_id,
        peft_model_revision=peft_model_revision,
        instruction_suffix=instruction_suffix,
        response_prefix=response_prefix,
        dtype=dtype,
    )

    generator = generators[generator_name](
        model=model,
        tokenizer=tokenizer,
        reward=reward_server,
        gen_args=gen_args_save,
        device=device,
        print_generation=print_generation,
    )

    dirname = os.path.dirname(output_filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with jsonlines.open(output_filename, "a") as fout:
        dataset_len = len(dataset)
        progress = tqdm(total=limit if limit > 0 else len(dataset))
        progress.update(skip_lines)
        for i in range(skip_lines, len(dataset), batch_size):
            if limit > 0 and i >= limit:
                print(f"{limit} limit!")
                break

            # 우선 아이템들을 인코딩합니다.
            end_i = min(dataset_len, i + batch_size)
            items = [dataset[j] for j in range(i, end_i)]
            items = generator(items, response_prefix, instruction_suffix)
            print(items)

            # 5. 결과 저장
            for item in items:
                r = item["response"]
                if additional_eos and additional_eos in r:
                    item["response"] = r.split(additional_eos, 1)[0]

                item["model_args"] = model_args
                item["gen_args"] = gen_args_save
                fout.write(item)

            progress.update(batch_size)


if __name__ == "__main__":
    fire.Fire(main)
