import os
import re
import json
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from think_utils import create_cot_message, questions, misconceptions

def remove_assessment(text):

    ASSESSMENT_RE = re.compile(r'^Assessment\:.*$', re.MULTILINE)

    return re.sub(ASSESSMENT_RE, '', text).strip()


def main(train_path:str):

    model_name = "Qwen/Qwen2.5-32B-Instruct-AWQ"

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize the vLLM engine
    llm = LLM(
        model=model_name,
        enable_prefix_caching=True,
        tensor_parallel_size=2, 
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
    )

    def prepare_correct_answers(train_data):
        idx = train_data.apply(lambda x: x.Category.split("_")[0] == 'True', axis=1)
        correct = train_data.loc[idx].copy()
        correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
        correct = correct.sort_values('c', ascending=False)
        correct = correct.drop_duplicates(['QuestionId'])[['QuestionId', 'MC_Answer']]
        correct['is_correct'] = True
        return correct

    df = pd.read_csv(train_path)
    df = df.drop_duplicates(subset=['QuestionId', 'QuestionText', 'MC_Answer', 'StudentExplanation']).reset_index(drop=True)
    df['QuestionId'] = df['QuestionId'].astype(str)
    df['Misconception'] = df['Misconception'].fillna("NA")

    correct = prepare_correct_answers(df)

    df = df.merge(correct, how='left')
    df['is_correct'] = df['is_correct'].fillna(False)

    inputs = []
    for r in df.itertuples():

        message = create_cot_message(r.QuestionId, questions, misconceptions, r.MC_Answer, r.StudentExplanation)

        input = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Set to False to strictly disable thinking
        )

        inputs.append(input)
        
    sampling_params = SamplingParams(
        temperature=0.6, 
        top_p=0.8,
        top_k=20,
        max_tokens=512,
    )

    # Generate outputs
    outputs = llm.generate(inputs, sampling_params)

    predictions = []

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text

        try:
            t = generated_text
        except Exception:
            t = ""
        predictions.append(t)


    df['cot'] = predictions
    df['cot'] = df['cot'].apply(lambda x: remove_assessment(x))

    df.to_parquet("./train_cot.parquet")

if __name__ == "__main__":

    main("../data/train.csv")