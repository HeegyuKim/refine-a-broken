from gradio_client import Client
import json
import torch
from tqdm import tqdm
from retry import retry


def batchify(items, batch_size):
    i = 0

    while i < len(items):
        yield items[i : i + batch_size]
        i += batch_size

    if i < len(items):
        yield items[i:]

class RewardAPI:
    def __init__(self, server_address: str, response_only: bool = False) -> None:
        self.client = Client(server_address)
        self.response_only = response_only

    @retry(tries=3, delay=5)
    def get_scores(self, conversations):
        if self.response_only:
            conversations = [
                [{
                    "role": "assistant",
                    "content": conv[-1]["content"],
                }]
                for conv in conversations
            ]
        result = self.client.predict(json.dumps(conversations), api_name="/predict")
        return json.loads(result)

    def get_scores_zip(self, instructions, responses, contexts=None):
        convs = []
        for i, (inst, res) in enumerate(zip(instructions, responses)):
            if contexts:
                conv = contexts[i].copy()
            else:
                conv = []
            conv.extend(
                [
                    {
                        "role": "user",
                        "content": inst,
                    },
                    {
                        "role": "assistant",
                        "content": res,
                    },
                ]
            )
            convs.append(conv)

        return self.get_scores(convs)
    
    def get_scores_items(self, items, responses):
        out_convs = []
        for i, (item, res) in enumerate(zip(items, responses)):
            out_convs.append(
                [
                    *item['conversations'],
                    {
                        "role": "assistant",
                        "content": res,
                    },
                ]
            )

        return self.get_scores(out_convs)
    
    def get_scores_conv(self, convs, responses):
        out_convs = []
        for i, (conv, res) in enumerate(zip(convs, responses)):
            out_convs.append(
                [
                    *conv,
                    {
                        "role": "assistant",
                        "content": res,
                    },
                ]
            )

        return self.get_scores(out_convs)
    

# if __name__ == "__main__":
#     server = RewardAPI("http://34.170.63.106:7860/")
#     print(server.get_scores([{"role": "user", "content": "Hello"},{"role": "assistant", "content": "What the hell"}]))
