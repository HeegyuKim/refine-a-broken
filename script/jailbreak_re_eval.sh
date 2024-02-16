
if [ $1 -eq 0 ]; then
    # change to the IP address of the Reward API server
    api_server="34.136.23.110"
elif [ $1 -eq 1 ]; then
    api_server="35.193.153.158"
fi


test_method() {
    model=$1
    python -m self_refine.re_eval_helpful \
        --input_dir "outputs/jailbreak-v2/$model/" \
        --output_dir "outputs/jailbreak-v2-helpful/$model" \
        --reward_api "http://$api_server:7860/"
}

if [ $1 -eq 0 ]; then
    test_method "meta-llama/Llama-2-7b-chat-hf"
    test_method "declare-lab/starling-7B"
elif [ $1 -eq 1 ]; then
    test_method "HuggingFaceH4/zephyr-7b-beta"
    test_method "berkeley-nest/Starling-LM-7B-alpha"
fi
