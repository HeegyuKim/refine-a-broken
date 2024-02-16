# refine-a-broken

This repository contains other repository as follows:
- [SafeRLHF](https://github.com/PKU-Alignment/safe-rlhf): Dai, Josef, et al. "Safe rlhf: Safe reinforcement learning from human feedback." arXiv preprint arXiv:2310.12773 (2023).
- [LLM-Attack](https://github.com/llm-attacks/llm-attacks): Zou, Andy, et al. "Universal and transferable adversarial attacks on aligned language models." arXiv preprint arXiv:2307.15043 (2023).
- [AutoDAN](https://github.com/SheltonLiu-N/AutoDAN): Liu, Xiaogeng, et al. "Autodan: Generating stealthy jailbreak prompts on aligned large language models." arXiv preprint arXiv:2310.04451 (2023).

## Setup

Install a proper version of PyTorch 2.1 that fits your hardware environments and  install other packages described in requirements.txt

```bash
pip install -r requirements.txt
```

## Reproduce Self-Refine
You can reproduce the self-refine process or download our generations from [link](https://drive.google.com/file/d/1nW7makmWRU5vc0eALWKf0pb2Jve4cuvy/view?usp=sharing) and decompress this file into `outputs/`.

Serve reward models for evaluating the helpfulness and cost
```
python -m safe_rlhf.gradio_reward2 \
    --model_type beaver \
    --model_id reward \
    --model2_id cost \
    --model2_type beaver \
    --device: cuda:0 \
    --batch_size 4 \
    --max_seq_length 2048 
```

Select model and baseline defense method for generation.

```
model_id="meta-llama/Llama-2-7b-chat-hf"
model_id="HuggingFaceH4/zephyr-7b-beta"
model_id="berkeley-nest/Starling-LM-7B-alpha"

# baseline defenses
generator_name="default" # no defense
generator_name="self-reminder"
generator_name="in-context-defense"
generator_name="smoothllm-v2"
```

Generate responses to jailbreak prompts combined with other harmful prompts
```
testset="jailbreak-targeting"
api_server="YOUR_IP_ADDRESS:PORT"
jailbreak="refusal_suppression,prefix_mwaha,prefix_evil,prefix_aim,suffix_here,mwaha+here,code_attack_mwaha"

python -m self_refine.generate \
    --model_id $model_id \
    --testset $testset \
    --generator_name $generator_name \
    --device cuda:0 \
    --dtype bfloat16 \
    --score_response_only \
    --reward_api "http://$api_server/" \
    --max_new_tokens 128 \
    --jailbreak "$jailbreak" \
    --batch_size $batch_size \
    --output_filename "./outputs/jailbreak-v2/$model_id/$generator_name/${testset}.json"
```

Generate response to jailbreak prompts searched by AutoDAN, GCG, other predefineds
```
# select jailbreak prompts
testset="model-gcg"
testset="model-autodan"
testset="jailbreak-non-targeting"

api_server="YOUR_IP_ADDRESS:PORT"

python -m self_refine.generate \
    --model_id $model_id \
    --testset $testset \
    --generator_name $generator_name \
    --device cuda:0 \
    --dtype bfloat16 \
    --score_response_only \
    --reward_api "http://$api_server/" \
    --max_new_tokens 128 \
    --batch_size $batch_size \
    --output_filename "./outputs/jailbreak-v2/$model_id/$generator_name/${testset}.json"
```


### Self-refine the harmful responses
```
# select formatting method
generator_name="self-refine-no-defense" # no formatting
generator_name="self-refine-defense" # code formatting
generator_name="self-refine-defense-json" # json formatting

api_server="YOUR_IP_ADDRESS:PORT"

input="generations to be self-refined"
output="A path to save self-refined responses"

python -m self_refine.generate_refine \
    --model_id $model_id \
    --device cuda:0 \
    --dtype bfloat16 \
    --generator_name $generator_name \
    --score_response_only \
    --reward_api "http://$api_server:7860/" \
    --max_new_tokens 128 \
    --batch_size $batch_size \
    --input_filename "$input" \
    --output_filename "$output"
```

## Searching Jailbreak Prompts using Automated Process
We utilized automated method to search jailbreak prompts for validating our study. Searched jailbreak prompts are saved in `data/` and we will delete these prompts when we publish our study for preventing adversarial misue. 

### llm-attack (GCG Attack)
we reproduced gcg attacks and added parsing code for searching best control for each goals

1. run llm-attack (check [this](https://github.com/llm-attacks/llm-attacks]))
2. parse llm-attack results to make the best control about goal
```
cd llm-attack/experiments
bash eval_scripts/log_parser.sh llama2 0 #individual 0 or multiple 1
```

### AutoDAN
Available models are described in [AutoDAN/models.py](./AutoDAN/models.py). Control the batch size according to your GPU environment.

```
cd AutoDAN

python autodan_hga_eval.py --model llama2 --batch_size 128
python autodan_hga_eval.py --model Starling-LM-7B-alpha --batch_size 128
python autodan_hga_eval.py --model zephyr --batch_size 128
```
