

export model=$1
export data=$2 # 0 for individual, 1 for transfer

if [ $data -eq 0 ];
then data='individual'
file_path="individual_${model}"
elif [ $data -eq 1 ];
then data='multiple'
file_path="transfer_${model}"
else echo "wrong data type"
exit 1
fi

  python -u log_parser.py \
        --attack_method="gcg" \
        --attack_data="${data}" \
        --model_id="${model}" \
        --file_path="${file_path}"