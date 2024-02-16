export CUDA_VISIBLE_DEVICES=$1
batch_size=4
testset="pku-saferlhf"
generator_name='self-refine-defense'

if [ $1 -eq 0 ]; then
    # change to the IP address of the Reward API server
    api_server="34.136.23.110"
elif [ $1 -eq 1 ]; then
    api_server="35.193.153.158"
fi

refine() {
    model_id=$1
    generator_name=$2
    input=$3
    output=$4

    python -m self_refine.generate_refine \
        --model_id $model_id \
        --testset $testset \
        --device cuda:0 \
        --dtype bfloat16 \
        --generator_name $generator_name \
        --score_response_only \
        --reward_api "http://$api_server:7860/" \
        --max_new_tokens 128 \
        --batch_size $batch_size \
        --input_filename "$input" \
        --output_filename "$output"
}
refine_model_steps() {
    model_id=$1
    generator_name=$2
    
    # first refinement
    refine "$model_id" "$generator_name" "./outputs/jailbreak-v2/$model_id/default/${testset}.json" "./outputs/jailbreak-v2/$model_id/default/${testset}-refine/$generator_name-1.json"

    # k-step refinement
    for k in {1..9}
    do
        input="$generator_name-$k"
        output="$generator_name-$((k + 1))"
        if [ -f "./outputs/jailbreak-v2/$model_id/default-v2/${testset}-refine/$input.json" ]; then
            refine "$model_id" "$generator_name" "./outputs/jailbreak-v2/$model_id/default/${testset}-refine/$input.json" "./outputs/jailbreak-v2/$model_id/default/${testset}-refine/$output.json"
        fi
    done
}

test_model() {
    model=$1
    generator_name=$2

    testset="jailbreak-targeting"
    refine_model_steps "$model" "$generator_name"
    testset="jailbreak-non-targeting"
    refine_model_steps "$model" "$generator_name"
    testset="model-gcg"
    refine_model_steps "$model" "$generator_name"
    testset="model-autodan"
    refine_model_steps "$model" "$generator_name"
}

test_method() {
    model=$1
    test_model "$model" "self-refine-defense"
    
    test_model "$model" "self-refine-no-defense"
    
    test_model "$model" "self-refine-defense-json"
}

if [ $1 -eq 0 ]; then
    test_method "meta-llama/Llama-2-7b-chat-hf"
    test_method "declare-lab/starling-7B"
elif [ $1 -eq 1 ]; then
    test_method "HuggingFaceH4/zephyr-7b-beta"
    test_method "berkeley-nest/Starling-LM-7B-alpha"
fi
