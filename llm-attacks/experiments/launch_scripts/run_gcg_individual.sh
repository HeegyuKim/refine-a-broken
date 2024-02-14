#!/bin/bash

#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

# export model=$1 # llama2 or vicuna
export setup=behaviors # behaviors or strings

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi


gen()
{
    export model=$1

 python -u ../main.py \
        --config="../configs/individual_${model}.py" \
        --config.attack=gcg \
        --config.train_data="../../data/advbench/harmful_${setup}.csv" \
        --config.result_prefix="../results/individual_${setup}_${model}_gcg_offset${data_offset}" \
        --config.n_train_data=10 \
        --config.data_offset=$data_offset \
        --config.n_steps=100 \
        --config.stop_on_success=False \
        --config.test_steps=20 \
        --config.batch_size=200

}

for data_offset in 0 10 20
do

    gen staring
    gen zephyr

done