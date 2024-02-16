
python -m safe_rlhf.gradio_reward2 \
    --model_type beaver \
    --model_id reward \
    --model2_id cost \
    --model2_type beaver \
    --device: cuda:0 \
    --batch_size 4 \
    --max_seq_length 2048 