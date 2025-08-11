"""
階層的分類用のユーティリティ関数
"""

import pandas as pd
import numpy as np
from transformers import DebertaV2Tokenizer
from datasets import Dataset
import torch
from sklearn.preprocessing import LabelEncoder


def prepare_correct_answers(train_data):
    """正解答案データを準備"""
    idx = train_data.apply(lambda row: row.Category.split('_')[0] == 'True', axis=1)
    correct = train_data.loc[idx].copy()
    correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
    correct = correct.sort_values('c', ascending=False)
    correct = correct.drop_duplicates(['QuestionId'])[['QuestionId','MC_Answer']]
    correct['is_correct'] = 1
    return correct


def format_input_hierarchical(row, task='category'):
    """階層的分類用の入力フォーマット"""
    if row['is_correct']:
        status = "This answer is correct."
    else:
        status = "This answer is incorrect."
    
    base_prompt = (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{status}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )
    
    if task == 'category':
        # Categoryを予測するタスク
        prompt = (
            f"{base_prompt}\n\n"
            f"Task: Classify if the explanation is correct, contains misconception, or neither."
        )
    else:
        # Misconceptionを予測するタスク
        prompt = (
            f"{base_prompt}\n\n"
            f"Task: Identify the specific misconception in the student's explanation."
        )
    
    return prompt


def tokenize_dataset(dataset, tokenizer, max_len):
    """データセットをトークナイズ"""
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=max_len)
    
    dataset = dataset.map(tokenize, batched=True)
    columns = ['input_ids', 'attention_mask', 'label'] if 'label' in dataset.column_names else ['input_ids', 'attention_mask']
    dataset.set_format(type='torch', columns=columns)
    return dataset


def compute_map3_category(eval_pred):
    """Category予測のためのaccuracy計算"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}


def compute_map3_misconception(eval_pred):
    """Misconception予測のためのMAP@3計算"""
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


def create_hierarchical_submission(category_predictions, misconception_predictions, 
                                 test_data, category_encoder, misconception_encoders):
    """階層的予測から提出ファイルを作成"""
    # Category予測の処理
    category_probs = torch.nn.functional.softmax(torch.tensor(category_predictions), dim=1).numpy()
    category_preds = np.argmax(category_probs, axis=1)
    category_labels = category_encoder.inverse_transform(category_preds)
    
    # 各行の予測を生成
    pred_strings = []
    for i, cat_label in enumerate(category_labels):
        if cat_label in ['True_Misconception', 'False_Misconception']:
            # Misconceptionがある場合
            encoder_key = cat_label
            if encoder_key in misconception_predictions and encoder_key in misconception_encoders:
                misc_probs = misconception_predictions[encoder_key][i]
                misc_probs_tensor = torch.nn.functional.softmax(torch.tensor(misc_probs), dim=0).numpy()
                top3_indices = np.argsort(-misc_probs_tensor)[:3]
                top3_misconceptions = misconception_encoders[encoder_key].inverse_transform(top3_indices)
                
                # 予測文字列を作成
                predictions = [f"{cat_label}:{misc}" for misc in top3_misconceptions]
            else:
                # フォールバック
                predictions = [f"{cat_label}:Incomplete", f"{cat_label}:Unknowable", f"{cat_label}:Irrelevant"]
        else:
            # MisconceptionがないCategory
            # MAP@3のため3つの予測を返す（同じものを3つ）
            predictions = [f"{cat_label}:NA"] * 3
        
        pred_strings.append(" ".join(predictions[:3]))  # 最大3つ
    
    submission = pd.DataFrame({
        'row_id': test_data.row_id.values,
        'Category:Misconception': pred_strings
    })
    return submission


def prepare_misconception_data(train_data, category):
    """特定のCategoryのMisconceptionデータを準備"""
    subset = train_data[train_data['Category'] == category].copy()
    
    # Misconceptionのエンコード
    le = LabelEncoder()
    subset['misconception_label'] = le.fit_transform(subset['Misconception'])
    
    return subset, le