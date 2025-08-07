"""
共通ユーティリティ関数
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset
import torch


def prepare_correct_answers(train_data):
    """正解答案データを準備"""
    idx = train_data.apply(lambda row: row.Category.split('_')[0] == 'True', axis=1)
    correct = train_data.loc[idx].copy()
    correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
    correct = correct.sort_values('c', ascending=False)
    correct = correct.drop_duplicates(['QuestionId'])[['QuestionId','MC_Answer']]
    correct['is_correct'] = 1
    return correct


def format_input(row, think:str = ""):
    """入力データをモデル用プロンプトにフォーマット"""
    if row["is_correct"]:
        status = "Yes"
    else:
        status = "No"

    # Phi-4用のプロンプトフォーマット（特別なthinkタグを含む）
    prompt = (
        "<|im_start|>system<|im_sep|>"
        "You are a math teacher grading students that took a diagnostic multiple choice math quuestion. "
        "You must classify the the explanation given by the student as to why they chose their answer.<|im_end|>"
        "<|im_start|>user<|im_sep|>\n"
        #f"[Mathematical Misconception Analysis Task]\n\n"
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct?: {status}\n"
        f"Explanation: {row['StudentExplanation']}<|im_end|>"
        "<|im_start|>assistant<|im_sep|>"
        f"<think>Let me analyze this mathematical misconception.\n{think}</think>\n\n"
    )
    return prompt


def tokenize_dataset(dataset, tokenizer, max_len):
    """データセットをトークナイズ"""
    def tokenize(batch):
        # パディングはDataCollatorで行うため、ここではトークナイズのみ
        return tokenizer(
            batch['text'],
            padding=False,  # パディングはDataCollatorに任せる
            truncation=True,
            max_length=max_len,
            return_tensors=None  # map時は'None'を使用
        )

    dataset = dataset.map(tokenize, batched=True, batch_size=100)
    # columnsの設定時にlabelを保持
    columns = ['input_ids', 'attention_mask', 'label'] if 'label' in dataset.column_names else ['input_ids', 'attention_mask']
    dataset.set_format(type='torch', columns=columns)
    return dataset


def compute_map3(eval_pred):
    """Top-3 予測に基づくMAP@3を計算"""
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


def compute_map3_with_reconstruction(eval_pred, is_correct_list, le_simplified):
    """
    Compute MAP@3 by reconstructing full labels from simplified predictions
    
    Args:
        eval_pred: (logits, simplified_labels) from trainer
        is_correct_list: List of boolean values indicating if answer was correct
        le_simplified: LabelEncoder for simplified labels
    """
    logits, simplified_labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    top3_simplified = np.argsort(-probs, axis=1)[:, :3]
    
    # Reconstruct full labels for ground truth
    true_full_labels = []
    for i, simplified_label in enumerate(simplified_labels):
        simplified_target = le_simplified.inverse_transform([simplified_label])[0]
        prefix = "True_" if is_correct_list[i] else "False_"
        full_target = prefix + simplified_target
        true_full_labels.append(full_target)
    
    # Reconstruct full labels for predictions
    score = 0.0
    for i, true_full_label in enumerate(true_full_labels):
        # Get top 3 simplified predictions and convert to full labels
        top3_full_predictions = []
        for simplified_idx in top3_simplified[i]:
            simplified_target = le_simplified.inverse_transform([simplified_idx])[0]
            prefix = "True_" if is_correct_list[i] else "False_"
            full_prediction = prefix + simplified_target
            top3_full_predictions.append(full_prediction)
        
        # Calculate MAP@3 score
        if true_full_label in top3_full_predictions:
            rank = top3_full_predictions.index(true_full_label) + 1
            score += 1.0 / rank
    
    return {"map@3": score / len(true_full_labels)}


def create_compute_metrics_function(is_correct_eval=None, le_simplified=None):
    def compute_metrics(eval_pred):
        if is_correct_eval is not None and le_simplified is not None:
            return compute_map3_with_reconstruction(
                eval_pred, 
                is_correct_eval, 
                le_simplified
            )
        else:
            # Fallback to simplified metric during training
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
    return compute_metrics


def create_submission(predictions, test_data, label_encoder):
    """予測結果から提出用ファイルを作成"""
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
    top3 = np.argsort(-probs, axis=1)[:, :3]
    flat = top3.flatten()
    decoded = label_encoder.inverse_transform(flat)
    top3_labels = decoded.reshape(top3.shape)
    pred_strings = [" ".join(r) for r in top3_labels]

    submission = pd.DataFrame({
        'row_id': test_data.row_id.values,
        'Category:Misconception': pred_strings
    })
    return submission