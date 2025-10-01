from prompt_utils import questions, question_explanations, misconception_explanations


def create_prompt_v1(tokenizer, row):
    if row["is_correct"]:
        status = "Yes"
    else:
        status = "No"

    # Create messages in the standard format
    messages = [
        {
            "role": "system",
            "content": (
                "You are a math teacher grading students that took a diagnostic multiple choice math question. "
                "You must classify the explanation given by the student as to why they chose their answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {row['QuestionText']}\n"
                f"Answer: {row['MC_Answer']}\n"
                f"Correct?: {status}\n"
                f"Explanation: {row['StudentExplanation']}"
            ),
        },
        {
            "role": "assistant",
            "content": "<think>Let me analyze this mathematical misconception.</think>\n\n",
        },
    ]

    # Apply the model's chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # Return string, not tokens
        add_generation_prompt=False,  # Don't add extra generation prompt since we have assistant message
    )

    return prompt


def create_prompt_think_v1(tokenizer, row):
    if row["is_correct"]:
        status = "Yes"
    else:
        status = "No"

    # Create messages in the standard format
    messages = [
        {
            "role": "system",
            "content": (
                "You are a math teacher grading students that took a diagnostic multiple choice math question. "
                "You must classify the explanation given by the student as to why they chose their answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {row['QuestionText']}\n"
                f"Answer: {row['MC_Answer']}\n"
                f"Correct?: {status}\n"
                f"Explanation: {row['StudentExplanation']}"
            ),
        },
        {
            "role": "assistant",
            "content": f"<think>Let me analyze this student's explanation. They said \"{row['StudentExplanation']}\".</think>\n\n",
        },
    ]

    # Apply the model's chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # Return string, not tokens
        add_generation_prompt=False,  # Don't add extra generation prompt since we have assistant message
    )

    return prompt


# Registering the function so that it can be called via configurations
prompt_registry = {
    "create_prompt_v1": create_prompt_v1,
    "create_prompt_think_v1": create_prompt_think_v1
}
