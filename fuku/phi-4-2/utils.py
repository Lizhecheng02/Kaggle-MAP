"""
共通ユーティリティ関数
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset
import torch
from typing import Any

# 設定からMAP@Kとメトリクス名を取得
from config import MAP_K, METRIC_NAME


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
    """入力データをモデル用プロンプトにフォーマット"""
    if row["is_correct"]:
        status = "Yes"
    else:
        status = "No"

    # Phi-4用のプロンプトフォーマット（特別なthinkタグを含む）
    # prompt = (
    #     "<|user|>\n"
    #     f"[Mathematical Misconception Analysis Task]\n\n"
    #     f"Question: {row['QuestionText']}\n"
    #     f"Answer: {row['MC_Answer']}\n"
    #     f"Correct?: {status}\n"
    #     f"Explanation: {row['StudentExplanation']}\n"
    #     "<|end|>\n"
    #     "<|assistant|>\n"
    #     "<think>\n"
    #     "Let me analyze this mathematical misconception...\n"
    #     "</think>\n\n"
    # )
    # prompt = (
    #     "<|im_start|>system<|im_sep|>"
    #     "You are a math teacher grading students that took a diagnostic multiple choice math question. "
    #     "You must classify the explanation given by the student as to why they chose their answer.<|im_end|>"
    #     "<|im_start|>user<|im_sep|>\n"
    #     f"Question: {row['QuestionText']}\n"
    #     f"Answer: {row['MC_Answer']}\n"
    #     f"Correct?: {status}\n"
    #     f"Explanation: {row['StudentExplanation']}<|im_end|>"
    #     "<|im_start|>assistant<|im_sep|>"
    # )


    # 一番いいプロンプト cv0.9481
    prompt = (
        "<|user|>\n"
        f"[Mathematical Misconception Analysis Task]\n\n"
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct?: {status}\n"
        f"Explanation: {row['StudentExplanation']}\n"
        "<|end|>\n"
        "<|assistant|>\n"
        "<think>\n"
        "Let me analyze this mathematical misconception...\n"
        "</think>\n\n"
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
    # columns の設定（階層タスクの追加ラベルも保持）
    base_cols = ['input_ids', 'attention_mask']
    extra = []
    for k in ['label', 'label_cat', 'label_mc']:
        if k in dataset.column_names:
            extra.append(k)
    columns = base_cols + extra
    dataset.set_format(type='torch', columns=columns)
    return dataset


def compute_map3(eval_pred: Any):
    """MAP@K を計算し、設定由来のキー名で返す。

    - Transformersの`EvalPrediction`でもタプル(preds, labels)でも動作。
    - torch依存を避け、numpyのみで安定計算。
    """
    # EvalPrediction 互換処理
    preds = getattr(eval_pred, "predictions", None)
    labels = getattr(eval_pred, "label_ids", None)
    if preds is None or labels is None:
        # タプル形式 (preds, labels) へのフォールバック
        try:
            preds, labels = eval_pred
        except Exception:
            raise ValueError("compute_map3: eval_pred から予測/ラベルを取得できませんでした")

    preds = np.asarray(preds)
    labels = np.asarray(labels)

    # softmax（数値安定化）
    x = preds - preds.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    probs = exp_x / exp_x.sum(axis=1, keepdims=True)

    k = int(MAP_K)
    topk = np.argpartition(-probs, kth=range(k), axis=1)[:, :k]
    # 各行についてソート（スコア順）
    row_indices = np.arange(probs.shape[0])[:, None]
    topk_sorted = topk[row_indices, np.argsort(-probs[row_indices, topk], axis=1)]

    score = 0.0
    for i, true_label in enumerate(labels):
        ranks = topk_sorted[i]
        for r, pred in enumerate(ranks, start=1):
            if pred == true_label:
                score += 1.0 / r
                break

    return {METRIC_NAME: float(score) / float(len(labels))}


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
