from prompt_utils import questions, question_explanations, misconception_explanations, question_label_choices, question_choices


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


def create_prompt_v2(tokenizer, row):
    if row["is_correct"]:
        status = "Yes"
    else:
        status = "No"

    choices = ""
    for c in question_choices[row['QuestionId']]:
        choices += f"• {c}\n"

    label_choices = ""
    for c in question_label_choices[row['QuestionId']]:
        label_choices += f"• {c}\n"

    misconception_text = ""
    for m in misconception_explanations[row['QuestionId']]:
        misconception_text += f"{m['Label']} ({m['LabelLong']}): {m['Explanation']}\n"

    # Create messages in the standard format
    messages = [
        {
            "role": "system",
            "content": (
                "You are a math teacher grading students that took a diagnostic multiple choice math question. "
                "You must classify the explanation given by the student as to why they chose their answer. "
                "When reading the student's explanation, ignore spelling and grammar errors. "
                "The choices range from \"A\", \"B\", \"C\", \"D\", but the choices are randomized for each student. "
                "Therefore if the student mentions an answer choice, ignore it. Focus on reasoning and content. "
                "Determine if the student gave an explanation that correctly explains how to answer the problem. "
                "Otherwise determine if the student fell into a common misconception associated with the question. "
                "Otherwise determine if the student's explanation is insufficient to conclude whether they understood how to solve the question or had a misconception."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {row['QuestionText']}\n"
                f"Question Choices:\n"
                f"{choices}\n"
                f"Student Answer: {row['MC_Answer']}\n"
                f"Student Answer Correct?: {status}\n"
                f"Common Misconceptions for Question:\n"
                f"{misconception_text}\n"
                f"Student Explanation: {row['StudentExplanation']}\n"
                f"Label Choice:\n"
                f"{label_choices}\n"
            ),
        },
        {
            "role": "assistant",
            "content": "<think>\n\n</think>\n\n",
        },
    ]

    # Apply the model's chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # Return string, not tokens
        add_generation_prompt=False,  # Don't add extra generation prompt since we have assistant message
    )

    return prompt


def create_prompt_gemma_v1(tokenizer, row):
    if row["is_correct"]:
        status = "Yes"
    else:
        status = "No"

    prompt = (
        f"<bos><start_of_turn>user\n"
        f"[Mathematical Misconception Analysis Task]\n\n"
        f"Question: {row['QuestionText']}\n"
        f"Student's Answer: {row['MC_Answer']}\n"
        f"Correct?: {status}\n"
        f"Student's Explanation: {row['StudentExplanation']}\n\n"
        # f"Misconception choices: {choices}\n\n"
        f"Let me analyze this mathematical misconception...\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    return prompt

# Registering the function so that it can be called via configurations
prompt_registry = {
    "create_prompt_v1": create_prompt_v1,
    "create_prompt_v2": create_prompt_v2,
    "create_prompt_gemma_v1": create_prompt_gemma_v1,
}



