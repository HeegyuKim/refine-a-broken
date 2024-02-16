"""
    streamlit run web_human_eval.py --server.port 7860
"""
import streamlit as st
import pandas as pd
import json, jsonlines, random
from datetime import datetime

num_pages = 5
with st.sidebar:
    page = st.radio("Select Page", 
             [str(i) for i in range(1, num_pages + 1)],
             key="page")
    page = int(page)

model2model = {
    "llama-2": "meta-llama__Llama-2-7b-chat-hf",
    "starling-7b-alpha": "berkeley-nest__Starling-LM-7B-alpha",
    "zephyr-7b-beta": "HuggingFaceH4__zephyr-7b-beta"
}
testsets = [
    "beavertails_eval-200", "harmfulqa_question-200"
]

def sort_by_question(items):
    items = sorted(items, key=lambda x: x["conversations"][-1]["content"])
    return items

name_A = "llama-2"
name_B = "starling-LM-7b-alpha"

def escape_markdown(text: str):
    """
    return text escaped given entity types
    :param text: text to escape
    :param entity_type: entities to escape
    :return:
    """
    # de-escape and escape again to avoid double escaping
    return text.replace('\\*', '*').replace('\\`', '`').replace('\\_', '_')\
        .replace('\\~', '~').replace('\\>', '>').replace('\\[', '[')\
        .replace('\\]', ']').replace('\\(', '(').replace('\\)', ')')\
        .replace('*', '\\*').replace('`', '\\`').replace('_', '\\_')\
        .replace('~', '\\~').replace('>', '\\>').replace('[', '\\[')\
        .replace(']', '\\]').replace('(', '\\(').replace(')', '\\)').replace('#', '\#').replace("\n", "\n\n")

def get_dataset():
    input_A = "../outputs/jailbreak-v2-humaneval/merged/Llama-2-7b-chat-hf.Self-Refine.json"
    input_B = "../outputs/jailbreak-v2-humaneval/merged/Starling-LM-7B-alpha.Self-Refine$_{code}$.json"
    input_A = sort_by_question(list(jsonlines.open(input_A)))
    input_B = sort_by_question(list(jsonlines.open(input_B)))
    return input_A, input_B

if "question_set" not in st.session_state:
    question_set = []

    input_A, input_B = get_dataset()

    assert len(input_A) == len(input_B)

    for i, A, B in zip(range(len(input_A)), input_A, input_B):
        assert A["conversations"][-1]["content"] == B["conversations"][-1]["content"]
        if random.random() > 0.5:
            item = {
                "A": A,
                "A_id": name_A,
                "B": B,
                "B_id": name_B,
            }
        else:
            item = { 
                "A": B,
                "A_id": name_B,
                "B": A,
                "B_id": name_A,
            }
        item['index'] = i + 1
        item['prompt'] = item['A']["conversations"][-1]["content"]
        question_set.append(item)

    st.session_state.question_set = question_set
else:
    question_set = st.session_state.question_set

items_per_page = len(question_set) // num_pages

def add_section(item):
    index = item["index"]

    st.divider()
    st.header(f"Q {item['index']}")
    st.markdown(escape_markdown(item["prompt"]))
    st.markdown("### Answer A")
    st.markdown(escape_markdown(item["A"]['response']))
    st.markdown("### Answer B")
    st.markdown(escape_markdown(item["B"]['response']))
    

    result = st.radio(
        "Which response is better between A and B?",
        ["A", "B", "Tie", "Both Bad"],
        key=f"topic-{index}"
        )
    
    if result == "A":
        result = item["A_id"]
    elif result == "B":
        result = item["B_id"]
    
    return {
        "result": {
            "select": result,
            **item
        }
    }

st.title("Jailbreak Human Evaluation")
st.subheader("This is a survey that evaluates which model is better by comparing the response generation results of the two models.")
st.text(f"Page = {page}")

results = []

for i in range((page-1)*items_per_page, min(len(question_set), page*items_per_page)):
    out = add_section(question_set[i])
    results.append(out)

st.divider()
st.success("Thank you for responding to the survey. Please click the download button to receive the result and send it to the authors")

if st.button("Show Results"):
    st.dataframe(pd.DataFrame(results).value_counts())

text_contents = "\n".join(json.dumps(x) for x in results)
filename = datetime.isoformat(datetime.now())
st.download_button(
    'Download Results', 
    text_contents, 
    file_name=f"{filename}-page{page}.json",
    on_click=lambda: st.balloons()
    )

# st.json(results)