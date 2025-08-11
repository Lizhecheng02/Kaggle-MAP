"""
共通ユーティリティ関数
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding as TransformersDataCollator
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


def format_input(row):
    """改善されたプロンプトフォーマット"""
    if row['is_correct']:
        status = "Yes"
    else:
        status = "No"

    # 誤概念を番号付きリストとして準備
    misconceptions = [
        'Adding_across', 'Adding_terms', 'Additive', 'Base_rate', 'Certainty',
        'Definition', 'Denominator-only_change', 'Division', 'Duplication',
        'Firstterm', 'FlipChange', 'Ignores_zeroes', 'Incomplete',
        'Incorrect_equivalent_fraction_addition', 'Interior', 'Inverse_operation',
        'Inversion', 'Irrelevant', 'Longer_is_bigger', 'Mult', 'Multiplying_by_4',
        'NA', 'Not_variable', 'Positive', 'Scale', 'Shorter_is_bigger',
        'Subtraction', 'SwapDividend', 'Tacking', 'Unknowable', 'WNB',
        'Whole_numbers_larger', 'Wrong_Fraction', 'Wrong_Operation',
        'Wrong_fraction', 'Wrong_term'
    ]

    # 番号付きリストを作成
    numbered_misconceptions = []
    for i, misc in enumerate(misconceptions, 1):
        numbered_misconceptions.append(f"{i}. {misc}")
    miscs_text = " | ".join(numbered_misconceptions)  # 1行にパック

    prompt = (
        f"<|im_start|>system\n"
        f"You are an expert math-education researcher.\n"
        f"Identify up to three *numbered* misconceptions from the allowed list that best match the student's explanation.\n"
        f"Respond with the numbers only, separated by commas, no other text.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Question: {row['QuestionText']}\n"
        f"Student answer (correct? {status}): {row['MC_Answer']}\n"
        f"Student explanation:\n"
        f"{row['StudentExplanation']}\n\n"
        f"Allowed misconceptions:\n"
        f"{miscs_text}\n\n"
        f"TASK: Return the IDs of the TOP 3 misconceptions (most probable first).\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
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


class DataCollatorWithPadding(TransformersDataCollator):
    """パディング付きデータコレーター"""

    def __init__(self, tokenizer, max_length=None):
        super().__init__(tokenizer=tokenizer)
        self.max_length = max_length

    def __call__(self, features):
        # 標準のDataCollatorWithPaddingの処理を実行
        batch = super().__call__(features)

        # max_lengthが指定されている場合は、その長さに切り詰める
        if self.max_length is not None:
            if 'input_ids' in batch:
                batch['input_ids'] = batch['input_ids'][:, :self.max_length]
            if 'attention_mask' in batch:
                batch['attention_mask'] = batch['attention_mask'][:, :self.max_length]

        return batch
