"""
Deberta Starter Python Script
このスクリプトは Kaggle の "MAP - Charting Student Math Misunderstandings" コンペティションで
Deberta モデルをトレーニングし、CV 0.93 を達成するためのものです。
"""

# --- Config ---
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

VER = 1
model_name = "/root/kaggle/map-charting-student-math-misunderstandings/models/deberta-v3-xsmall"
EPOCHS = 10
DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)

# --- Load Train Data ---
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category + ":" + train.Misconception
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")
print(train.head())

# --- Powerful Feature Engineering ---
idx = train.apply(lambda row: row.Category.split('_')[0] == 'True', axis=1)
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)

# --- EDA: Display Question and Answer Choices ---
# (Kaggle ノートブックと同様に実行するときは IPython.display を利用してください)

tmp = train.groupby(['QuestionId','MC_Answer']).size().reset_index(name='count')
tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
tmp = tmp.drop('count', axis=1).sort_values(['QuestionId','rank'])
questions = tmp.QuestionId.unique()
for q in questions:
    question_text = train.loc[train.QuestionId==q].iloc[0].QuestionText
    choices = tmp.loc[tmp.QuestionId==q].MC_Answer.values
    labels = "ABCD"
    choice_str = " ".join([f"({labels[i]}) {c}" for i,c in enumerate(choices)])
    print(f"QuestionId {q}: {question_text}")
    print(f"MC Answers: {choice_str}\n")

# --- Imports for Training ---
import torch
from transformers import DebertaV2ForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import matplotlib.pyplot as plt

# --- Tokenizer Setup ---
from transformers import DebertaV2Tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
MAX_LEN = 256

# --- Format Input Function ---
def format_input(row):
    """入力データをモデル用プロンプトにフォーマットします"""
    if row['is_correct']:
        status = "This answer is correct."
    else:
        status = "This answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{status}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

train['text'] = train.apply(format_input, axis=1)
print("Example prompt for our LLM:")
print(train.text.values[0])

# --- Token Length Distribution ---
lengths = [len(tokenizer.encode(t, truncation=False)) for t in train['text']]
plt.hist(lengths, bins=50)
plt.title("Token Length Distribution")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

over_limit = (np.array(lengths) > MAX_LEN).sum()
print(f"There are {over_limit} train sample(s) with more than {MAX_LEN} tokens")

# --- Create Validation Subset ---
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)
COLS = ['text','label']
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds = Dataset.from_pandas(val_df[COLS])

# --- Tokenize Datasets ---
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=MAX_LEN)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)

# --- Initialize Model ---
model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=200,
    eval_steps=200,
    save_total_limit=1,
    metric_for_best_model="map@3",
    greater_is_better=True,
    load_best_model_at_end=True,
    report_to="none",
)

# --- Custom MAP@3 Metric ---
def compute_map3(eval_pred):
    """Top-3 予測に基づくMAP@3を計算します"""
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    top3 = np.argsort(-probs, axis=1)[:, :3]
    score = 0.0
    for i, label in enumerate(labels):
        ranks = top3[i]
        if ranks[0] == label:
            score += 1.0
        elif ranks[1] == label:
            score += 1.0 / 2
        elif ranks[2] == label:
            score += 1.0 / 3
    return {"map@3": score / len(labels)}

# --- Trainer Setup and Training ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_map3,
)
trainer.train()

# --- Save Model and Encoder ---
trainer.save_model(f"{DIR}/best")
import joblib
joblib.dump(le, f"{DIR}/label_encoder.joblib")

# --- Inference on Test Set ---

test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)
test['text'] = test.apply(format_input, axis=1)

ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)

predictions = trainer.predict(ds_test)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()

# --- Create Submission ---
top3 = np.argsort(-probs, axis=1)[:, :3]
flat = top3.flatten()
decoded = le.inverse_transform(flat)
top3_labels = decoded.reshape(top3.shape)
pred_strings = [" ".join(r) for r in top3_labels]

import pandas as pd2
sub = pd2.DataFrame({
    'row_id': test.row_id.values,
    'Category:Misconception': pred_strings
})
sub.to_csv('submission.csv', index=False)
print(sub.head())
