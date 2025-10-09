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


def rewrite_questions(df, rewrite_dict):
    """指定されたQuestionIdのQuestionTextを書き換え"""
    for question_id, new_text in rewrite_dict.items():
        df.loc[df['QuestionId'] == question_id, 'QuestionText'] = new_text
    return df


def replace_wrong_fraction(row):
    """Wrong_FractionとWrong_fractionを統一する。QuestionId 33471の行のみ適用"""
    if row["QuestionId"] == 33471 and row["MC_Answer"] == "Wrong_fraction":
        row["MC_Answer"] = "Wrong_Fraction"
    return row

def wrong_fraction_decoder(
    submission: pd.DataFrame,
    test_data: pd.DataFrame | None = None,
    *,
    from_label: str | None = None,
    to_label: str | None = None,
    question_ids: list[int] | None = None,
    apply_globally: bool | None = None,
) -> pd.DataFrame:
    """
    予測提出データのラベル表記ゆれを修正するデコーダ。

    - 既知の不一致: "Wrong_Fraction" → "Wrong_fraction"
    - 既定では特定の `QuestionId`(設定ファイルの `WRONG_FRACTION_FIX_QIDS`) に対してのみ置換する。
    - `apply_globally=True` の場合は全行に対して置換する。

    Parameters
    ----------
    submission : pd.DataFrame
        列 `row_id`, `Category:Misconception` を持つ提出用データフレーム。
    test_data : pd.DataFrame | None
        列 `row_id`, `QuestionId` を含むテストデータ。行ごとの `QuestionId` を特定するために使用。
    from_label : str | None
        置換元ラベル。未指定時は設定値 `WRONG_FRACTION_FROM_LABEL` を使用。
    to_label : str | None
        置換先ラベル。未指定時は設定値 `WRONG_FRACTION_TO_LABEL` を使用。
    question_ids : list[int] | None
        置換対象の `QuestionId` リスト。未指定時は設定値 `WRONG_FRACTION_FIX_QIDS` を使用。
    apply_globally : bool | None
        True の場合は全行置換。未指定時は設定値 `WRONG_FRACTION_APPLY_GLOBALLY` を使用。

    Returns
    -------
    pd.DataFrame
        置換後の新しい DataFrame（元の `submission` は変更しない）。
    """

    # ユーザー要望により設定は関数内でハードコード
    # 呼び出し時に引数が指定されればそれを優先
    DEFAULT_FROM = "Wrong_Fraction"
    DEFAULT_TO = "Wrong_fraction"
    DEFAULT_QIDS = [33471]
    DEFAULT_APPLY_GLOBALLY = False

    from_label = from_label or DEFAULT_FROM
    to_label = to_label or DEFAULT_TO
    question_ids = question_ids or list(DEFAULT_QIDS)
    apply_globally = DEFAULT_APPLY_GLOBALLY if apply_globally is None else apply_globally

    if "Category:Misconception" not in submission.columns:
        raise ValueError("submissionに'Category:Misconception'列が存在しません。")
    if "row_id" not in submission.columns:
        raise ValueError("submissionに'row_id'列が存在しません。")

    # 置換ロジック
    def _replace_line(s: str) -> str:
        # 3つのラベルが空白区切りで並ぶ前提を保ちつつ部分置換
        parts = (s or "").split()
        return " ".join([to_label if p == from_label else p for p in parts])

    out = submission.copy()

    if apply_globally:
        out["Category:Misconception"] = out["Category:Misconception"].map(_replace_line)
        return out

    # 行ごとにQuestionIdに基づいて制限
    if test_data is None:
        # test_data が無い場合は何もしない（安全側）。
        return out

    if "row_id" not in test_data.columns or "QuestionId" not in test_data.columns:
        # 必要列が無ければ何もしない（安全側）。
        return out

    qid_map = (
        test_data[["row_id", "QuestionId"]]
        .drop_duplicates()
        .set_index("row_id")["QuestionId"]
    )
    out["__qid__"] = out["row_id"].map(qid_map)
    mask = out["__qid__"].isin(set(question_ids))
    out.loc[mask, "Category:Misconception"] = (
        out.loc[mask, "Category:Misconception"].map(_replace_line)
    )
    out = out.drop(columns=["__qid__"])  # 一時列を削除
    return out
