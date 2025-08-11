import os
os.environ["VLLM_USE_V1"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import vllm
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import (
    MODEL_PATH, BATCH_SIZE, SYS_PROMPT_MISCONCEPTION,
    load_and_preprocess_data, add_correct_answer_features,
    create_misconception_prompt, parse_llm_response, 
    predict_batch, create_submission_predictions, compute_map3
)

# Load and preprocess data
trn, test, le, misconception_counts = load_and_preprocess_data()

# Add correct answer features
trn, test = add_correct_answer_features(trn, test)

# Show top misconceptions
print(f"\nTop 10 misconceptions:")
print(misconception_counts.head(10))

# Initialize LLM
print("\nInitializing LLM...")
llm = vllm.LLM(
    MODEL_PATH,
    quantization='awq',
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=3072,
    disable_log_stats=True,
    enable_prefix_caching=True,
)
tokenizer = llm.get_tokenizer()

# Create train/validation split
print("\nCreating train/validation split...")
# Filter out rare targets for stratification
target_counts = trn['target'].value_counts()
valid_targets = target_counts[target_counts >= 2].index
trn_filtered = trn[trn['target'].isin(valid_targets)]

if len(trn_filtered) < len(trn):
    print(f"Filtered {len(trn) - len(trn_filtered)} samples with rare targets")

train_df, val_df = train_test_split(trn_filtered, test_size=0.15, random_state=42, stratify=trn_filtered['target'])
print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

# Generate predictions for validation set
print("\nGenerating predictions for validation set...")
val_prompts = val_df.apply(create_misconception_prompt, axis=1).tolist()
val_predictions = []

# Process in batches
for i in tqdm(range(0, len(val_prompts), BATCH_SIZE)):
    batch_prompts = val_prompts[i:i+BATCH_SIZE]
    batch_responses = predict_batch(llm, tokenizer, batch_prompts, SYS_PROMPT_MISCONCEPTION)
    batch_predictions = [parse_llm_response(resp) for resp in batch_responses]
    val_predictions.extend(batch_predictions)

# Calculate validation MAP@3
print("\nCalculating validation MAP@3...")
val_true = val_df['target'].tolist()

# Create probability matrix with smoothing
unique_targets = le.classes_
target_to_idx = {target: idx for idx, target in enumerate(unique_targets)}
n_classes = len(unique_targets)
n_val = len(val_predictions)
val_probs = np.ones((n_val, n_classes)) * 0.001  # Small probability for all classes

# Assign higher probabilities based on predictions and common patterns
for i, pred in enumerate(val_predictions):
    if pred in target_to_idx:
        val_probs[i, target_to_idx[pred]] = 0.7
        
        # Add probability to similar predictions
        category = pred.split(":")[0]
        misconception = pred.split(":")[1]
        
        # Similar categories
        for target, idx in target_to_idx.items():
            if category in target:
                val_probs[i, idx] += 0.1
            if misconception != "NA" and misconception in target:
                val_probs[i, idx] += 0.1
    
    # Normalize probabilities
    val_probs[i] = val_probs[i] / val_probs[i].sum()

val_map3 = compute_map3(val_true, val_probs, le)
print(f"Validation MAP@3: {val_map3:.4f}")

# Generate predictions for test set
print("\nGenerating predictions for test set...")
test_prompts = test.apply(create_misconception_prompt, axis=1).tolist()
test_predictions = []

# Process in batches
for i in tqdm(range(0, len(test_prompts), BATCH_SIZE)):
    batch_prompts = test_prompts[i:i+BATCH_SIZE]
    batch_responses = predict_batch(llm, tokenizer, batch_prompts, SYS_PROMPT_MISCONCEPTION)
    batch_predictions = [parse_llm_response(resp) for resp in batch_responses]
    test_predictions.extend(batch_predictions)

# Create submission predictions
print("\nCreating submission file...")
submission_predictions = create_submission_predictions(test_predictions, test, misconception_counts, le)

# Create submission dataframe
submission = pd.DataFrame({
    'row_id': test['row_id'],
    'Category:Misconception': submission_predictions
})

# Save submission
submission.to_csv('submission.csv', index=False)
print("\nSubmission saved to submission.csv")
print(f"Submission shape: {submission.shape}")
print("\nFirst 5 predictions:")
print(submission.head())

# Show sample predictions with explanations
print("\n\nSample test predictions with explanations:")
for i in range(min(3, len(test))):
    print(f"\n--- Sample {i+1} ---")
    print(f"Question: {test.iloc[i]['QuestionText'][:100]}...")
    print(f"Answer: {test.iloc[i]['MC_Answer']} ({'Correct' if test.iloc[i]['is_correct'] else 'Incorrect'})")
    print(f"Explanation: {test.iloc[i]['StudentExplanation'][:100]}...")
    print(f"Predictions: {submission.iloc[i]['Category:Misconception']}")