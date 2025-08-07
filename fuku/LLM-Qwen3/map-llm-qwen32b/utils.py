import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Configuration
MODEL_PATH = "/kaggle/input/qwen-3-32b-awq"
BATCH_SIZE = 8
TEMPERATURE = 0.1
TOP_P = 0.95
MAX_TOKENS = 256

# System prompt for misconception detection
SYS_PROMPT_MISCONCEPTION = """You are an expert educational psychologist analyzing student math explanations.

Given a question, student's answer, and their explanation, identify:
1. Category: Combines answer correctness with explanation quality
   - True_Correct: Correct answer with correct reasoning
   - True_Misconception: Correct answer but explanation shows misconception
   - True_Neither: Correct answer with unclear/incomplete explanation
   - False_Correct: Incorrect answer but correct reasoning process
   - False_Misconception: Incorrect answer with identifiable misconception
   - False_Neither: Incorrect answer with unclear reasoning
   
2. Misconception: The specific misconception type if present, otherwise "NA"

Common misconceptions:
- Incomplete: Missing important steps
- Additive: Incorrectly adding when other operations needed
- Duplication: Repeating elements incorrectly
- Subtraction: Errors in subtraction
- Positive: Assuming all numbers must be positive
- Wrong_term: Using incorrect mathematical terms
- Irrelevant: Unrelated explanations
- Wrong_fraction: Errors in fraction operations
- Inversion: Reversing operations/relationships
- Mult: Multiplication errors

Respond in format:
Category: [category]
Misconception: [type or NA]"""


def load_and_preprocess_data(train_path="/kaggle/input/map-charting-student-math-misunderstandings/train.csv", 
                             test_path="/kaggle/input/map-charting-student-math-misunderstandings/test.csv"):
    """Load and preprocess train and test data"""
    print("Loading data...")
    trn = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print("Preprocessing data...")
    trn['Misconception'] = trn['Misconception'].fillna('NA')
    trn['target'] = trn['Category'] + ":" + trn['Misconception']
    
    # Create label encoder and get unique misconceptions
    le = LabelEncoder()
    le.fit(trn['target'])
    n_classes = len(le.classes_)
    print(f"Number of target classes: {n_classes}")
    
    # Get misconception statistics
    misconception_counts = trn[trn['Misconception'] != 'NA']['Misconception'].value_counts()
    
    return trn, test, le, misconception_counts


def add_correct_answer_features(trn, test):
    """Add correct answer features to train and test data"""
    # Feature engineering: identify correct answers
    idx = trn.apply(lambda row: row.Category.split('_')[0] == 'True', axis=1)
    correct = trn.loc[idx].copy()
    correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
    correct = correct.sort_values('c', ascending=False)
    correct = correct.drop_duplicates(['QuestionId'])[['QuestionId','MC_Answer']]
    correct['is_correct'] = 1
    
    # Merge correct answer information
    if 'is_correct' not in trn.columns:
        trn = trn.merge(correct, on=['QuestionId','MC_Answer'], how='left')
        trn['is_correct'] = trn['is_correct'].fillna(0).astype(int)
    
    test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    test['is_correct'] = test['is_correct'].fillna(0).astype(int)
    
    return trn, test


def create_misconception_prompt(row):
    """Create prompt for misconception detection"""
    status = "correct" if row['is_correct'] == 1 else "incorrect"
    
    prompt = f"""Question: {row['QuestionText']}
Student's Answer: {row['MC_Answer']} (This answer is {status})
Student's Explanation: {row['StudentExplanation']}

Analyze the explanation and identify the category and any misconception."""
    
    return prompt


def parse_llm_response(response):
    """Parse LLM response to extract category and misconception"""
    category = "False_Neither"
    misconception = "NA"
    
    # Try to extract category
    category_match = re.search(r'Category:\s*([^\n]+)', response, re.IGNORECASE)
    if category_match:
        category = category_match.group(1).strip()
    
    # Try to extract misconception
    misconception_match = re.search(r'Misconception:\s*([^\n]+)', response, re.IGNORECASE)
    if misconception_match:
        misconception = misconception_match.group(1).strip()
    
    # Validate category
    valid_categories = ["True_Correct", "True_Misconception", "True_Neither", 
                       "False_Correct", "False_Misconception", "False_Neither"]
    if category not in valid_categories:
        category = "False_Neither"
    
    # Clean misconception
    if misconception.lower() in ["na", "n/a", "none", "no misconception"]:
        misconception = "NA"
    
    return f"{category}:{misconception}"


def predict_batch(llm, tokenizer, prompts, sys_prompt):
    """Generate predictions for a batch of prompts"""
    import vllm
    
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


def create_submission_predictions(test_predictions, test, misconception_counts, le):
    """Create submission predictions with fallback logic"""
    submission_predictions = []
    
    # Get top misconceptions for fallback predictions
    top_misconceptions = misconception_counts.head(5).index.tolist()
    target_to_idx = {target: idx for idx, target in enumerate(le.classes_)}
    
    for i, pred in enumerate(test_predictions):
        row = test.iloc[i]
        predictions = []
        
        # First prediction from model
        if pred in target_to_idx:
            predictions.append(pred)
        else:
            # Default based on answer correctness
            if row['is_correct'] == 1:
                predictions.append("True_Correct:NA")
            else:
                predictions.append("False_Neither:NA")
        
        # Add diverse second and third predictions
        if row['is_correct'] == 1:
            if "True_Correct" not in predictions[0]:
                predictions.append("True_Correct:NA")
            predictions.append("True_Neither:NA")
            if len(predictions) < 3:
                predictions.append("True_Misconception:Incomplete")
        else:
            # For incorrect answers, predict common misconceptions
            if "False_Misconception" not in predictions[0]:
                predictions.append(f"False_Misconception:{top_misconceptions[0]}")
            predictions.append("False_Neither:NA")
            if len(predictions) < 3:
                predictions.append(f"False_Misconception:{top_misconceptions[1]}")
        
        # Ensure we have exactly 3 predictions
        predictions = predictions[:3]
        submission_predictions.append(" ".join(predictions))
    
    return submission_predictions


def compute_map3(y_true, y_pred_probs, label_encoder):
    """Compute MAP@3 score"""
    score = 0.0
    n_samples = len(y_true)
    
    for i in range(n_samples):
        # Get top 3 predictions
        top3_indices = np.argsort(-y_pred_probs[i])[:3]
        top3_labels = label_encoder.inverse_transform(top3_indices)
        
        # Check if true label is in top 3
        true_label = y_true[i]
        for rank, pred_label in enumerate(top3_labels):
            if pred_label == true_label:
                score += 1.0 / (rank + 1)
                break
    
    return score / n_samples