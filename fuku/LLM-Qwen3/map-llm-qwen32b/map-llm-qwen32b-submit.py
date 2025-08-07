import os
os.environ["VLLM_USE_V1"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import vllm
import torch
from tqdm import tqdm
from utils import (
    MODEL_PATH, BATCH_SIZE, SYS_PROMPT_MISCONCEPTION,
    load_and_preprocess_data, add_correct_answer_features,
    create_misconception_prompt, parse_llm_response, 
    predict_batch, create_submission_predictions
)

# Load and preprocess data
trn, test, le, misconception_counts = load_and_preprocess_data()

# Add correct answer features
trn, test = add_correct_answer_features(trn, test)

# Show top misconceptions
print(f"\nTop 5 misconceptions:")
print(misconception_counts.head(5))

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