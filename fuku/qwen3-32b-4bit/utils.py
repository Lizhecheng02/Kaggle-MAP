"""
共通ユーティリティ関数
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset
import torch
from config import (
    TARGET_BASECATEGORY_COL,
    TARGET_QUESTION_ID_COL,
    TARGET_MISCONCEPTION_COL,
    TARGET_NEITHER_KEYWORD,
    TARGET_NEITHER_PREFIX,
    TARGET_BASECATEGORY_FALLBACK_COLS,
    SUBMISSION_USE_QUESTION_LABEL_CHOICES,
    QUESTION_LABEL_CHOICES,
)


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
    prompt = (
        "<|im_start|>user"
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
    """予測結果から提出用ファイルを作成

    - 設定で SUBMISSION_USE_QUESTION_LABEL_CHOICES が有効な場合、QuestionIdごとの候補ラベルに制約。
    - そうでない場合（または対応候補が無い場合）、全クラスから上位3件を採用。
    - test_data に is_correct 列があれば "True_" / "False_" を接頭辞として付与。
    """

    logits = np.asarray(predictions.predictions)

    # 質問ごとの候補ラベルを、LabelEncoderのインデックスへ変換
    choice_ids_map = {}
    if SUBMISSION_USE_QUESTION_LABEL_CHOICES:
        classes = np.asarray(label_encoder.classes_)
        for qid, choices in QUESTION_LABEL_CHOICES.items():
            mask = np.isin(classes, choices)
            idx = np.where(mask)[0]
            choice_ids_map[str(qid)] = [int(x) for x in idx]

    test_probabilities = []
    test_predictions = []
    test_top3_predictions = []

    has_correct = 'is_correct' in test_data.columns

    for qid, row_logits, correct in zip(
        test_data['QuestionId'].astype(str).tolist(),
        logits,
        test_data['is_correct'].tolist() if has_correct else [False] * len(test_data),
    ):
        # 候補インデックス（なければ全クラスを対象に）
        candidate_idx = choice_ids_map.get(qid)
        if not candidate_idx:  # None or empty
            candidate_idx = list(range(logits.shape[1]))

        candidate_logits = np.asarray(row_logits)[candidate_idx]
        candidate_probs = torch.nn.functional.softmax(torch.tensor(candidate_logits), dim=-1).numpy()
        top_k_local = np.argsort(-candidate_probs)[:3]

        # グローバルクラス空間に戻す
        topk_idx = np.asarray(candidate_idx)[top_k_local]
        topk_probs = candidate_probs[top_k_local].tolist()
        topk_labels = label_encoder.inverse_transform(topk_idx).tolist()

        # 出力ラベルのフォーマット: True_/False_ + (QuestionIdやQを除いた部分)
        correct_prefix = "True_" if bool(correct) else "False_"
        formatted = []
        for lab in topk_labels:
            # "31772:Correct:NA" -> ["31772", "Correct:NA"]
            parts = lab.split(":", maxsplit=1)
            tail = parts[1] if len(parts) > 1 else lab
            formatted.append(correct_prefix + tail)

        test_probabilities.append(topk_probs)
        test_predictions.append(formatted)
        test_top3_predictions.append(" ".join(formatted[:3]))

    submission = pd.DataFrame({
        "row_id": test_data["row_id"].tolist() if 'row_id' in test_data.columns else list(range(len(test_data))),
        "QuestionId": test_data["QuestionId"].tolist(),
        "is_correct": test_data["is_correct"].tolist() if has_correct else [False] * len(test_data),
        "probs": test_probabilities,
        "preds": test_predictions,
        'Category:Misconception': test_top3_predictions,
    })

    return submission


def process_targets(row: pd.Series) -> str:
    """ターゲットラベルの文字列を作成

    仕様:
      - BaseCategory に TARGET_NEITHER_KEYWORD を含む場合:
          f"{TARGET_NEITHER_PREFIX}:{BaseCategory}:{Misconception}"
      - それ以外:
          f"{QuestionId}:{BaseCategory}:{Misconception}"

    すべてのカラム名・キーワードは config.py の設定値に依存します。
    """
    # BaseCategory列のフォールバック対応
    base_col = None
    if TARGET_BASECATEGORY_COL in row.index:
        base_col = TARGET_BASECATEGORY_COL
    else:
        for c in TARGET_BASECATEGORY_FALLBACK_COLS:
            if c in row.index:
                base_col = c
                break
    if base_col is None:
        raise KeyError(
            f"BaseCategory-like column not found. Expected one of: {[TARGET_BASECATEGORY_COL, *TARGET_BASECATEGORY_FALLBACK_COLS]}"
        )

    base = row[base_col]
    misc = row[TARGET_MISCONCEPTION_COL]
    if isinstance(base, str) and TARGET_NEITHER_KEYWORD in base:
        return f"{TARGET_NEITHER_PREFIX}:{base}:{misc}"
    else:
        qid = row[TARGET_QUESTION_ID_COL]
        return f"{qid}:{base}:{misc}"
