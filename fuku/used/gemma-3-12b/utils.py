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

    # 正誤を自然文にする（Gemmaの事前学習分布に寄せる）
    status = "This answer is correct." if row['is_correct'] else "This answer is incorrect."

    # 選択肢となる誤概念リストを明示（モデルにラベル空間を意識させる）
    misconceptions = [
        'Adding_across', 'Adding_terms', 'Additive', 'Base_rate', 'Certainty', 'Definition',
        'Denominator-only_change', 'Division', 'Duplication', 'Firstterm', 'FlipChange',
        'Ignores_zeroes', 'Incomplete', 'Incorrect_equivalent_fraction_addition', 'Interior',
        'Inverse_operation', 'Inversion', 'Irrelevant', 'Longer_is_bigger', 'Mult',
        'Multiplying_by_4', 'NA', 'Not_variable', 'Positive', 'Scale', 'Shorter_is_bigger',
        'Subtraction', 'SwapDividend', 'Tacking', 'Unknowable', 'WNB', 'Whole_numbers_larger',
        'Wrong_Fraction', 'Wrong_Operation', 'Wrong_fraction', 'Wrong_term'
    ]
    choices = ", ".join(misconceptions)

    # Gemma-3用のプロンプト（chatテンプレート + 選択肢）
    prompt = (
        f"<bos><start_of_turn>user\n"
        f"[Mathematical Misconception Analysis Task]\n\n"
        f"Question: {row['QuestionText']}\n"
        f"Student's Answer: {row['MC_Answer']}\n"
        f"Status: {status}\n"
        f"Student's Explanation: {row['StudentExplanation']}\n\n"
        f"Misconception choices: {choices}\n\n"
        f"Task: Identify the mathematical misconception in this student's reasoning from the above list."
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    # if row["is_correct"]:
    #     status = "Yes"
    # else:
    #     status = "No"
    # prompt = (
    #     f"<bos><start_of_turn>user\n"
    #     f"[Mathematical Misconception Analysis Task]\n\n"
    #     f"Question: {row['QuestionText']}\n"
    #     f"Answer: {row['MC_Answer']}\n"
    #     f"Correct?: {status}\n"
    #     f"Explanation: {row['StudentExplanation']}\n"
    #     f"Task: Identify the mathematical misconception in this student's reasoning."
    #     f"<end_of_turn>\n"
    #     f"<start_of_turn>model\n"
    # )
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


def get_last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    attention_maskから各サンプルの最終有効トークン位置（padを除いた最後のインデックス）を返す。
    形状:
      - attention_mask: (batch, seq_len)
      - return: (batch,) のlongテンソル
    """
    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must be 2D tensor (batch, seq_len)")
    # sum-1 で最後の有効トークン位置
    idx = attention_mask.long().sum(dim=1) - 1
    # 負値を0に丸める安全策（全ゼロのケースを考慮）
    idx = torch.clamp(idx, min=0)
    return idx


def pool_last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    (batch, seq_len, hidden) のhidden_statesとattention_maskから、
    各サンプルの最後の有効トークンの隠れ状態をプーリングして (batch, hidden) を返す。
    """
    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must be 3D tensor (batch, seq_len, hidden)")
    if attention_mask.shape[0] != hidden_states.shape[0] or attention_mask.shape[1] != hidden_states.shape[1]:
        raise ValueError("attention_mask and hidden_states must have matching batch and seq_len dimensions")

    idx = get_last_token_indices(attention_mask)
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    pooled = hidden_states[batch_indices, idx]
    return pooled


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
