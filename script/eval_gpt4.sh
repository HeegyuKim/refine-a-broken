
eval() {
    python -m self_refine.eval_gpt4 \
        --input_A "$1" \
        --input_B "$2" \
        --temperature 0 \
        --output_filename "$3" 
}

eval_datasets() {
    modelA=$1
    modelB=$2
    output=$3

    modelA="outputs/jailbreak-v2-humaneval/merged/$modelA.json"
    modelB="outputs/jailbreak-v2-humaneval/merged/$modelB.json"
    output="outputs/jailbreak-v2-humaneval/gpt4/$output.json"
    eval "$modelA" "$modelB" "$output"
}

llama="Llama-2-7b-chat-hf.Self-Refine"
starling_alpha_code="Starling-LM-7B-alpha.Self-Refine\$_{code}$"
starling_alpha_json="Starling-LM-7B-alpha.Self-Refine\$_{json}$"


eval_datasets "$llama" "$starling_alpha_json" "llama2_vs_starling-alpha-json"
eval_datasets "$llama" "$starling_alpha_code" "llama2_vs_starling-alpha-code"
eval_datasets "$starling_alpha_code" "$starling_alpha_json" "starling-alpha-code_vs_starling-alpha-json"