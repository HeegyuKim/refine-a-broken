from safe_rlhf.api_server import HfArgumentParser, RewardServerArgument, RewardServer, InferenceRequest, MultiRewardServerArgument

import gradio as gr
import json


parser = HfArgumentParser([MultiRewardServerArgument])
args = parser.parse_args_into_dataclasses()[0]
args2 = args.get_model2_args()
server1 = RewardServer(args)
server2 = RewardServer(args2)

def get_rewards(json_text):
    items = json.loads(json_text)
    if isinstance(items, dict):
        items = [[items]]
    elif not isinstance(items[0], list):
        items = [items]
    print(items)
    output1 = server1.get_score(InferenceRequest(conversations=items))
    output2 = server2.get_score(InferenceRequest(conversations=items))
    
    output = {}
    output[args.model1_name] = output1
    output[args.model2_name] = output2

    return json.dumps(output, ensure_ascii=False)

examples = [
    ['[{"role": "user", "content": "Hello"},{"role": "assistant", "content": "Hi"}]'],
    ['[{"role": "user", "content": "Hello"},{"role": "assistant", "content": "What the hell"}]'],
]
if __name__ == "__main__":

    demo = gr.Interface(
        fn=get_rewards,
        inputs=["text"],
        outputs=[gr.Text({}, label="json",)],
        examples=examples
    )
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
