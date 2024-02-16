model=$1
python autodan_ga_eval.py --model $model --batch_size 128
python autodan_hga_eval.py --model $model --batch_size 128
