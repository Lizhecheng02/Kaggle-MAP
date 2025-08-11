import os
os.environ["VLLM_USE_V1"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import vllm
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Configuration
model_path = "/kaggle/input/qwen-3-32b-awq"
BATCH_SIZE = 16
TEMPERATURE = 0.1
TOP_P = 0.9
MAX_TOKENS = 512

# Load data
print("Loading data...")
trn = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

# Preprocess data
trn['Misconception'] = trn['Misconception'].fillna('NA')
trn['target'] = trn['Category'] + ":" + trn['Misconception']

# Create label encoder
le = LabelEncoder()
le.fit(trn['target'])
n_classes = len(le.classes_)
print(f"Number of target classes: {n_classes}")

# Feature engineering: identify correct answers
idx = trn.apply(lambda row: row.Category.split('_')[0] == 'True', axis=1)
correct = trn.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

# Merge correct answer information to test
test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test['is_correct'] = test['is_correct'].fillna(0)

# Initialize LLM
print("Initializing LLM...")
llm = vllm.LLM(
    model_path,
    quantization='awq',
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=4096,
    disable_log_stats=True,
    enable_prefix_caching=True,
)
tokenizer = llm.get_tokenizer()

# System prompt for misconception detection
SYS_PROMPT_MISCONCEPTION = """You are an expert educational psychologist specializing in mathematics education. Your task is to analyze student explanations for math questions and identify potential misconceptions.

Given a math question, the student's multiple-choice answer, and their explanation, you need to:
1. Determine if the answer is correct or incorrect
2. Identify if the explanation contains a misconception, is correct, or neither
3. If there's a misconception, identify the specific type

Common misconceptions include:
- Incomplete: Missing important steps or partial understanding
- Additive: Incorrectly adding when other operations are needed
- Duplication: Repeating elements incorrectly
- Subtraction: Errors in subtraction operations
- Positive: Assuming all numbers must be positive
- Wrong_term: Using incorrect mathematical terms
- Irrelevant: Providing unrelated explanations
- Wrong_fraction: Errors in fraction operations
- Inversion: Reversing operations or relationships
- Mult: Errors in multiplication

Format your response EXACTLY as:
Category: [True_Correct|True_Misconception|True_Neither|False_Correct|False_Misconception|False_Neither]
Misconception: [specific misconception type or NA]
"""

def create_misconception_prompt(row):
    """Create prompt for misconception detection"""
    status = "correct" if row['is_correct'] else "incorrect"
    
    prompt = f"""Question: {row['QuestionText']}

Student's Answer: {row['MC_Answer']}
This answer is {status}.

Student's Explanation: {row['StudentExplanation']}

Analyze the student's explanation and identify:
1. The appropriate category based on answer correctness and explanation quality
2. Any specific misconception present (or NA if none)

Remember:
- True_Correct: Correct answer with correct explanation
- True_Misconception: Correct answer but explanation shows misconception
- True_Neither: Correct answer but unclear/incomplete explanation
- False_Correct: Incorrect answer but correct reasoning (rare)
- False_Misconception: Incorrect answer with identifiable misconception
- False_Neither: Incorrect answer with unclear reasoning
"""
    return prompt

def parse_llm_response(response):
    """Parse LLM response to extract category and misconception"""
    lines = response.strip().split('\n')
    category = "False_Neither"  # default
    misconception = "NA"  # default
    
    for line in lines:
        if line.startswith("Category:"):
            category = line.split(":", 1)[1].strip()
        elif line.startswith("Misconception:"):
            misconception = line.split(":", 1)[1].strip()
    
    # Validate category
    valid_categories = ["True_Correct", "True_Misconception", "True_Neither", 
                       "False_Correct", "False_Misconception", "False_Neither"]
    if category not in valid_categories:
        category = "False_Neither"
    
    # Build target
    target = f"{category}:{misconception}"
    return target

def predict_batch(llm, tokenizer, prompts, sys_prompt):
    """Generate predictions for a batch of prompts"""
    messages_batch = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        messages_batch.append(formatted_prompt)
    
    outputs = llm.generate(
        messages_batch,
        vllm.SamplingParams(
            seed=42,
            skip_special_tokens=True,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P
        ),
    )
    
    responses = [output.outputs[0].text.strip() for output in outputs]
    return responses

# Generate predictions for test set
print("Generating predictions for test set...")
test_prompts = test.apply(create_misconception_prompt, axis=1).tolist()
test_predictions = []

# Process in batches
for i in tqdm(range(0, len(test_prompts), BATCH_SIZE)):
    batch_prompts = test_prompts[i:i+BATCH_SIZE]
    batch_responses = predict_batch(llm, tokenizer, batch_prompts, SYS_PROMPT_MISCONCEPTION)
    batch_predictions = [parse_llm_response(resp) for resp in batch_responses]
    test_predictions.extend(batch_predictions)

# Create submission
print("Creating submission file...")
submission_predictions = []
target_to_idx = {target: idx for idx, target in enumerate(le.classes_)}

for pred in test_predictions:
    if pred in target_to_idx:
        # Get top prediction
        top1 = pred
        # Get 2 more diverse predictions based on common patterns
        if "True_Correct" in pred:
            top2 = "True_Neither:NA"
            top3 = "False_Neither:NA"
        elif "False_Misconception" in pred:
            misconception = pred.split(":")[1]
            top2 = "False_Neither:NA"
            top3 = f"False_Misconception:Incomplete"
        else:
            top2 = "False_Neither:NA"
            top3 = "True_Correct:NA"
        
        submission_predictions.append(f"{top1} {top2} {top3}")
    else:
        # Default prediction
        submission_predictions.append("False_Neither:NA True_Correct:NA False_Misconception:Incomplete")

submission = pd.DataFrame({
    'row_id': test['row_id'],
    'Category:Misconception': submission_predictions
})

submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")
print(f"Submission shape: {submission.shape}")
print("\nFirst 5 predictions:")
print(submission.head())