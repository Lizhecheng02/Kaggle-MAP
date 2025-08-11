import pandas as pd

from sklearn.model_selection import StratifiedKFold

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from utils import prepare_correct_answers
from prompts import (
    create_question_cot_prompt,
    create_question_think_user_prompt,
    create_question_think_assistant_output,
)

from dotenv import dotenv_values

CONFIG = dotenv_values(".env")


def parse_cot_to_think(output: str) -> dict:
    try:
        problem_solution_explanation = (
            output.split("</problem_solution_explanation>")[0]
            .split("<problem_solution_explanation>")[1]
            .strip()
        )
        evaluation = output.split("<evaluation>")[1].split("</evaluation>")[0].strip()
    except Exception:
        problem_solution_explanation = ""
        evaluation = ""

    return {
        "think": f"Let's first solve this problem: {problem_solution_explanation} {evaluation}"
    }


def main(train_path: str = "../data/train.csv", output_path: str = "../data"):
    """
    Creating dataset to distill <think> from a LLM with high performance in
    reasoning (Gemini-2.5-Flash) into a smaller model (Phi-4-reasoning-plus).
    """

    # Load data
    trn = pd.read_csv(train_path)
    trn["Misconception"] = trn["Misconception"].fillna("NA")

    # Identify correct answers for the quesstion
    correct = prepare_correct_answers(trn)
    trn = trn.merge(correct, on=["QuestionId", "MC_Answer"], how="left")
    trn.is_correct = trn.is_correct.fillna(0)

    # Deduplicate similar text
    trn["FilteredStudentExplanation"] = trn["StudentExplanation"].apply(
        lambda x: x.lower().strip().strip(".").strip()
    )
    trn = trn.drop_duplicates("FilteredStudentExplanation", keep="first")
    del trn["FilteredStudentExplanation"]

    # Create Chain-of-Thought prompts
    # Instead of directly creating content for <think> we will first decompose the problem into:
    # 1) An explanation of how to solve the given math problem
    # 2) An evaluation of the student's explanation given all the information available to us
    cot_prompts = [
        create_question_cot_prompt(str(r.QuestionId), r.MC_Answer, r.StudentExplanation)
        for r in trn.itertuples()
    ]

    # Access Gemini API with API Key
    api_key = CONFIG["GEMINI_API_KEY"]
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)

    # Set configurations
    # Using the recommended parameters as per Google
    generation_config = GenerationConfig(
        temperature=1.0, candidate_count=1, top_p=0.95, top_k=64
    )

    model_name = "gemini-2.5-flash"

    # Using Gemini 2.5 Flash
    model = genai.GenerativeModel(
        model_name=model_name, generation_config=generation_config
    )

    # Generate the content and store the output
    # This is not efficient - use async method for parallel requests
    cot_gemini_2_5_flash = []
    for prompt in cot_prompts:
        try:
            output = model.generate_content(prompt)
            cot_gemini_2_5_flash.append(output.text)

        except Exception:
            cot_gemini_2_5_flash.append("")

    # Parse the <problem_solution_explanation> and <evaluation> tags
    think_gemini_2_5_flash = [parse_cot_to_think(x) for x in cot_gemini_2_5_flash]

    # Create a dataframe with the processed <think> content
    think_gemini_2_5_flash_df = pd.DataFrame(think_gemini_2_5_flash)

    if len(trn) != len(think_gemini_2_5_flash_df):
        raise ValueError(
            "Length of generated Chain of Thought does not match length of DataFrame"
        )

    trn["model"] = model_name
    trn["think"] = think_gemini_2_5_flash_df["think"]

    # Use this to later create user prompts to generate <think>
    user_prompts = [
        create_question_think_user_prompt(
            str(r.QuestionId), r.MC_Answer, r.StudentExplanation
        )
        for r in trn.itertuples()
    ]

    assistant_outputs = [
        create_question_think_assistant_output(
            str(r.QuestionId), r.Category, r.Misconception, r.is_correct, r.think
        )
        for r in trn.itertuples()
    ]

    trn["user"] = user_prompts
    trn["assistant"] = assistant_outputs

    # Create folds stratifying against the label
    # Will create 10 folds
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    trn["label"] = trn["Category"] + ":" + trn["Misconception"]
    trn["fold"] = 0
    for i, (trn_idx, val_idx) in enumerate(skf.split(trn, trn["label"], trn["label"])):
        trn.loc[val_idx, "fold"] = i

    trn.to_parquet(f"{output_path}/think_prompts_{model_name}_fold.parquet")


if __name__ == "__main__":
    main()
