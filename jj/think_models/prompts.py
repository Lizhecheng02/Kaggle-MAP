from questions import questions
from misconceptions import misconception_explanations


def create_question_cot_prompt(
    qid: str, student_answer: str, student_explanation: str
) -> str:
    # Creating the problem data
    correct_choice = [x["Choice"] for x in questions[qid]["Choices"] if x["Correct"]][0]
    problem_data = f"Question: {questions[qid]['QuestionText']}\nChoices:\n"
    for choice in questions[qid]["Choices"]:
        problem_data += f"• {choice['Choice']}\n"
    problem_data += f"Correct Choice: {correct_choice}"

    # Creating the misconception data
    misconception_data = "\n".join(
        [
            f"{x['LabelLong']}: {x['Explanation']}"
            for x in misconception_explanations[qid]
        ]
    ).strip()

    prompt = f"""You will analyze a diagnostic multiple choice math problem, the student's answer, and the student's explanation for why they chose their answer.
Your goal is to explain precisely whether the student's explanation demonstrates a clear understanding of how to solve the problem, or whether the student's explanation demonstrates a common known misconception associated with the problem, or whether the student's explanation was insufficient.

Here is the problem information:
<problem_data>
{problem_data}
</problem_data>

Here are common misconceptions associated with this particular problem:
<common_misconceptions>
{misconception_data}
</common_misconceptions>

Here is the answer that the student chose for the problem:
<student_answer>
{student_answer}
</student_answer>

Here is the explanation that the student gave for why they chose their answer:
<student_explanation>
{student_explanation}
</student_explanation>

First, solve the math problem and provide a simple explanation for how to solve it
- Your answer must be the same as the "Correct Choice"
- Your explanation must be brief and simple

Write your explanation in <problem_solution_explanation> tags.

Then examine all components of the problem carefully:
1. The problem statement and question asked
2. The given problem choices
3. The correct answer choice for the probelm
4. The known misconceptions associated with the problem
5. The explanation that you gave to solve the problem from your <problem_solution_explanation>
6. The answer that the student chose
7. The explanation that the student gave

Then, give an evaluation of the student's explanation
- Determine if the student's explanation correctly and sufficiently explains how to solve the problem
- Determine if the student's explanation demonstrates one of the common misconceptions associated with the problem
- Determine if the student's explanation is neither correct nor demonstrates a misconception
- Keep your explanation to 5-6 clear, non-repetitive sentences

Write your evaluation in <evaluation> tags.

Guidelines for writing your explanation:
- Do not restate the problem
- Do not restate the student's explanation
- Do not quote the student's explanation
- Avoid repetition
- Stay focused with evaluating the student's explanation"""

    return prompt


def create_question_think_user_prompt(
    qid: str, student_answer: str, student_explanation: str
) -> str:
    """
    Creating the user prompt to generate the <think> content.
    """

    # Creating the problem data
    correct_choice = [x["Choice"] for x in questions[qid]["Choices"] if x["Correct"]][0]
    problem_data = f"Question: {questions[qid]['QuestionText']}\nChoices:\n"
    for choice in questions[qid]["Choices"]:
        problem_data += f"• {choice['Choice']}\n"
    problem_data += f"Correct Choice: {correct_choice}"

    # Creating the misconception data
    misconception_data = "\n".join(
        [
            f"{x['LabelLong']}: {x['Explanation']}"
            for x in misconception_explanations[qid]
        ]
    ).strip()

    prompt = f"""You will analyze a diagnostic multiple choice math problem, the student's answer, and the student's explanation for why they chose their answer.
Your goal is to explain precisely whether the student's explanation demonstrates a clear understanding of how to solve the problem, or whether the student's explanation demonstrates a common known misconception associated with the problem, or whether the student's explanation was insufficient.

Here is the problem information:
<problem_data>
{problem_data}
</problem_data>

Here are common misconceptions associated with this particular problem:
<common_misconceptions>
{misconception_data}
</common_misconceptions>

Here is the answer that the student chose for the problem:
<student_answer>
{student_answer}
</student_answer>

Here is the explanation that the student gave for why they chose their answer:
<student_explanation>
{student_explanation}
</student_explanation>

First, solve the math problem and provide a simple explanation for how to solve it
- Your answer must be the same as the "Correct Choice"
- Your explanation must be brief and simple

Then examine all components of the problem carefully:
1. The problem statement and question asked
2. The given problem choices
3. The correct answer choice for the probelm
4. The known misconceptions associated with the problem
5. The explanation that you gave to solve the problem from your <problem_solution_explanation>
6. The answer that the student chose
7. The explanation that the student gave

Then, give an evaluation of the student's explanation
- Determine if the student's explanation correctly and sufficiently explains how to solve the problem
- Determine if the student's explanation demonstrates one of the common misconceptions associated with the problem
- If the explanation has a misconception, then provide it, otherwise return "Not Applicable"
- Determine if the student's explanation is neither correct nor demonstrates a misconception

Return just the results of your evaluation in the following format:

<"True"|"False"> ("True" if student chose correct answer otherwise "False")
<"Correct"|"Misconception"|"Neither">
<Applicable Misconception|"Not Applicable">
"""

    return prompt


def create_question_think_assistant_output(
    qid: str, category: str, misconception: str, is_correct: bool, think: str
) -> str:
    """
    Creating the output expected to be generated by the assistant.
    """
    a = "True" if is_correct else "False"

    c = category.split("_")[1]

    m = "Not Applicable"
    for misconception_explanation in misconception_explanations[qid]:
        if misconception_explanation["Label"] == misconception:
            m = misconception_explanation["LabelLong"]

    assistant = f"""<think>{think}</think>

{a}
{c}
{m}
"""

    return assistant
