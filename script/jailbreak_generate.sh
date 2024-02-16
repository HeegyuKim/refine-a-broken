export CUDA_VISIBLE_DEVICES=$1
batch_size=4

# dataset (jailbreak-targeting, jailbreak-non-targeting)
testset="jailbreak-targeting"

# - default (no defense)
# - self-refine-defense
# - self-reminder
# - smoothllm
generator_name="default"

if [ $1 -eq 0 ]; then
    # change to the IP address of the Reward API server
    api_server="34.136.23.110"
elif [ $1 -eq 1 ]; then
    api_server="35.193.153.158"
fi

gen() {
    model_id=$1

    if [ $testset == "jailbreak-targeting" ]; then
        jailbreak="refusal_suppression,prefix_mwaha,prefix_evil,prefix_aim,suffix_here,mwaha+here,code_attack_mwaha"
    else
        jailbreak=""
    fi

    python -m self_refine.generate \
        --model_id $model_id \
        --testset $testset \
        --generator_name $generator_name \
        --device cuda:0 \
        --dtype bfloat16 \
        --score_response_only \
        --reward_api "http://$api_server:7860/" \
        --max_new_tokens 128 \
        --jailbreak "$jailbreak" \
        --batch_size $batch_size \
        --output_filename "./outputs/jailbreak-v2/$model_id/$generator_name/${testset}.json"
}

gen_testset() {
    model_id=$1
    testset="model-gcg"
    gen $model_id
    testset="model-autodan"
    gen $model_id
    testset="jailbreak-targeting"
    gen $model_id
    testset="jailbreak-non-targeting"
    gen $model_id
}

gen_method() {
    model_id=$1
    generator_name="default"
    gen_testset $model_id

    generator_name="self-reminder"
    gen_testset $model_id

    generator_name="in-context-defense"
    gen_testset $model_id

    generator_name="smoothllm-v2"
    gen_testset $model_id
}

if [ $1 -eq 0 ]; then
    gen_method "meta-llama/Llama-2-7b-chat-hf"
    gen_method "declare-lab/starling-7B"
elif [ $1 -eq 1 ]; then
    gen_method "HuggingFaceH4/zephyr-7b-beta"
    gen_method "berkeley-nest/Starling-LM-7B-alpha"
fi