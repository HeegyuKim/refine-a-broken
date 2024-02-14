# refine-a-broken






# llm-attack
we reproduce gcg attack data with [repogitory](https://github.com/llm-attacks/llm-attacks) made by Authors of ["Universal and Transferable Adversarial Attacks on Aligned Language Models"](https://arxiv.org/abs/2307.15043)
and only add parse code for find best control for each goals

1. run llm-attack (check [this](https://github.com/llm-attacks/llm-attacks]))
2. parse llm-attack results to make the best control about goal
```cd llm-attack/experiments
bash eval_scripts/log_parser.sh llama2 0 #individual 0 or multiple 1
```
