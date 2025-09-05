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


def format_input(row):
    """入力データをモデル用プロンプトにフォーマット"""
    if row["is_correct"]:
        status = "Yes"
    else:
        status = "No"

    # Qwen2.5-Math用の数学タスクに特化したプロンプト
    # prompt = (
    #     "<|im_start|>user"
    #     f"[Mathematical Misconception Analysis Task]\n\n"
    #     f"Question: {row['QuestionText']}\n"
    #     f"Answer: {row['MC_Answer']}\n"
    #     f"Correct?: {status}\n"
    #     f"Explanation: {row['StudentExplanation']}\n\n"
    #     "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    # )
    prompt = (
        "<|im_start|>system\n"
        "You are a math teacher grading students that took a diagnostic multiple choice math question. "
        "You must classify the explanation given by the student as to why they chose their answer.<|im_end|>\n"
        "<|im_start|>user\n"
        f"[Mathematical Misconception Analysis Task]\n\n"
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct?: {status}\n"
        f"Explanation: {row['StudentExplanation']}\n\n"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
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





def update_question_texts(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """
    DataFrame内の特定のQuestionIdに対してQuestionTextを一括置換します。

    Parameters:
    - df: 対象のDataFrame（'QuestionId', 'QuestionText'列が必要）
    - context: ログ出力用の任意の文字列（例: "train", "submit"）

    Returns:
    - QuestionTextが更新されたDataFrame（in-placeで更新しつつ同じdfを返す）
    """

    updates = {
        32835: "Which number is the greatest? Options: 6.0000 6.2 6.079 6.0001",
        31772: "There is a large equilateral triangle. It has been divided into 9 smaller equilateral triangles of the same size. Six of these are colored. What fraction of the whole is the uncolored portion? Write your answer as an irreducible fraction."
    }

    # 必要列がなければスキップ
    required = {"QuestionId", "QuestionText"}
    if not required.issubset(df.columns):
        print(f"update_question_texts: 必要列 {required} が見つかりません。スキップします。")
        return df

    # ログ（任意）
    for qid, new_text in updates.items():
        mask = df['QuestionId'] == qid
        cnt = int(mask.sum())
        tag = f" ({context})" if context else ""
        if cnt > 0:
            original = df.loc[mask, 'QuestionText'].iloc[0]
            print(f"QuestionId {qid}: {cnt}件ヒット{tag}")
            print(f"  元: {original[:80]}...")
            print(f"  新: {new_text}")
        else:
            print(f"QuestionId {qid}: 該当なし{tag}")

    # 一括置換
    mapped = df['QuestionId'].map(updates)
    mask_update = mapped.notna()
    df.loc[mask_update, 'QuestionText'] = mapped[mask_update]
    prefix = f"{context}: " if context else ""
    print(f"{prefix}合計 {int(mask_update.sum())} 行を更新しました。")

    return df
