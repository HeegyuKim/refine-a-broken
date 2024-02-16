import uvicorn
from fastapi import FastAPI
from threading import Lock
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForSequenceClassification
from safe_rlhf.models import AutoModelForScore
import numpy as np
import torch
import importlib


from pydantic import BaseModel
from dataclasses import dataclass
from typing import Optional, List, Union, Dict
import urllib, requests
from collections import defaultdict


class RewardModel:
    def __init__(self, model_id: str, max_seq_length: int, **kwargs):
        self.model_id = model_id
        self.max_seq_length = max_seq_length

    def get_score(self, conversations):
        pass

    def format_conversations(self, conversations):
        pass

    
class BeaverModel(RewardModel):
    def __init__(self, model_id: str, max_seq_length: int, device: str, **kwargs):
        super().__init__(model_id, max_seq_length)
        if model_id == "reward":
            repo_id = "PKU-Alignment/beaver-7b-v1.0-reward"
        
        elif model_id == "cost":
            repo_id = "PKU-Alignment/beaver-7b-v1.0-cost"
        
        self.device = device
        self.model = AutoModelForScore.from_pretrained(repo_id).eval().half()
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=False)

        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

    def format_conversations(self, conversations):
        out_convs = []
        for convs in conversations:
            lines = [
                "BEGINNING OF CONVERSATION:"
            ]
            for conv in convs:
                if conv["role"] == "user":
                    lines.append("USER: " + conv["content"])
                else:
                    lines.append("ASSISTANT: " + conv["content"])
            
            out_convs.append("\n".join(lines))
        return out_convs

    @torch.no_grad()
    def get_score(self, conversations):
        
        input_ids = self.tokenizer(
            self.format_conversations(conversations), 
            return_tensors='pt',
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length"
            ).to(self.device)
        output = self.model(**input_ids)
        return output.end_scores.squeeze(-1).tolist()

MODEL_TYPES = {
    "beaver": BeaverModel
}

@dataclass
class RewardServerArgument:
    model_type: str = "beaver"
    model_id: str = "reward"
    revision: Optional[str] = None
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    host: str = "0.0.0.0"
    port: int = 7860
    batch_size: int = 4
    max_seq_length: int = 1024

@dataclass
class MultiRewardServerArgument(RewardServerArgument):
    model2_type: str = "beaver"
    model2_id: str = "cost"
    revision: Optional[str] = None
    model1_name: str = "helpful"
    model2_name: str = "safety"

    def get_model2_args(self):
        return RewardServerArgument (
            model_type=self.model2_type,
            model_id=self.model2_id,
            revision=self.revision,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
        )


class InferenceRequest(BaseModel):
    conversations: List[List[Dict]] = None



def batched(iterator, batch_size):
    batch = []
    for example in iterator:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


class RewardServer:
    def __init__(self, args: RewardServerArgument) -> None:
        self.device = args.device
        self.args = args
        self.lock = Lock()

        self.prepare_model(args)

    def prepare_model(self, args: RewardServerArgument):
        print("loading model")
        model_type = MODEL_TYPES[args.model_type]
        self.model = model_type(
            args.model_id, 
            args.max_seq_length,
            device=self.device,
            revision=args.revision
            )
        
        print("caching xla model")
        conv = [[{"role": "user", "content": "dummy"}]] * args.batch_size
        self.get_score(InferenceRequest(conversations=conv))

    def serve_ready(self):
        return self.args.model_id

    def get_score(self, request: InferenceRequest):
        outputs = defaultdict(list)
        batched_iterator = list(batched(request.conversations, self.args.batch_size))
        # with self.lock:
        for batch_conversations in batched_iterator:
            dummy_size = self.args.batch_size - len(batch_conversations)
            if dummy_size > 0:
                dummies = [[{"role": "user", "content": "dummy"}]] * dummy_size
                batch_conversations.extend(dummies)

            scores = self.model.get_score(batch_conversations)

            if dummy_size > 0:
                scores = scores[:-dummy_size]

            outputs["scores"].extend(scores)
        
        return outputs
        
    def get_score_iter(self, conversations):
        batched_iterator = list(batched(conversations, self.args.batch_size))
        # with self.lock:
        for batch_conversations in batched_iterator:
            dummy_size = self.args.batch_size - len(batch_conversations)
            if dummy_size > 0:
                dummies = [[{"role": "user", "content": "dummy"}]] * dummy_size
                batch_conversations.extend(dummies)

            scores = self.model.get_score(batch_conversations)

            if dummy_size > 0:
                scores = scores[:len(batch_conversations)]

            yield from scores        
    
    
    def run(self):
        app = FastAPI()
        app.get("/ready")(self.serve_ready)
        app.get("/score")(self.get_score)
        uvicorn.run(app, host=self.args.host, port=self.args.port)


class RewardClient(object):

    def __init__(self, url: str):
        self.url = url

    def get_scores(self, conversations: List[List[Dict]]):
        response = requests.get(
            urllib.parse.urljoin(self.url, "score"),
            json={"conversations": conversations},
        ).json()

        return response
        
    def get_scores_batch_iter(self, conversations: List[List[Dict]], batch_size: int = 4):
        batched_iterator = list(batched(conversations, batch_size))
        for batch_conversations in batched_iterator:
            response = requests.get(
                urllib.parse.urljoin(self.url, "score"),
                json={"conversations": batch_conversations},
            ).json()

            for i in range(len(batch_conversations)):
                yield {k: response[k][i] for k in response}


    