from tqdm.auto import tqdm
from glob import glob
import jsonlines as jl
import os
from .rm_utils import RewardAPI
from fire import Fire


def main(
    input_dir: str,
    output_dir: str,
    reward_api: str,
    batch_size: int = 32
):
    files = glob(os.path.join(input_dir, "**/*.json"), recursive=True)
    api = RewardAPI(reward_api, response_only=False) # "http://34.32.223.200:7860/")

    for file in files:
        output_file = os.path.join(output_dir, file.replace(input_dir, ""))
        with jl.open(file, "r") as fin:
            items = list(fin)

        for i in tqdm(range(0, len(items), batch_size), desc=f"Processing {file}"):
            end_i = min(i+batch_size, len(items))
            prompts = items
            responses = [x['response'] for x in items[i:end_i]]

            res_scores = api.get_scores_items(prompts, responses)

            for item, help, safe in zip(items[i:end_i], res_scores["helpful"]['scores'], res_scores["safety"]['scores']):
                # item['score-safety'] = safe
                # helpful swap only
                item['score-helpful'] = help

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with jl.open(output_file, "w") as fout:
            fout.write_all(items)


if __name__ == "__main__":
    Fire(main)