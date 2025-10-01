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


def _build_label_instruction(label_names):
    """ラベル列挙のインストラクション文を作成"""
    if not label_names:
        return ""
    label_list = " | ".join(map(str, label_names))
    return (
        "\nTask: Classify the student's explanation into one of the valid labels.\n"
        f"Valid labels ({len(label_names)}): {label_list}\n"
        "Output: one exact label string from the list above.\n\n"
    )


def build_fewshot_examples_map(df, per_label=1, max_examples_per_q=3):
    """
    各QuestionIdに対するfew-shot例を構築
    - 代表例として、同一QuestionId内で頻度の高いtargetごとに最大per_label件を抽出
    - 例は最大max_examples_per_q件に抑制
    返り値: { QuestionId: [ {idx, MC_Answer, is_correct, StudentExplanation, target}, ... ] }
    """
    # 必要列の存在チェックと補完
    df = df.copy()
    if 'target' not in df.columns and {'Category', 'Misconception'} <= set(df.columns):
        df['Misconception'] = df['Misconception'].fillna('NA')
        df['target'] = df['Category'].astype(str) + ':' + df['Misconception'].astype(str)

    if 'is_correct' not in df.columns and {'QuestionId', 'MC_Answer'} <= set(df.columns):
        # 訓練データから正答フラグを推定
        correct = prepare_correct_answers(df)
        df = df.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
        df['is_correct'] = df['is_correct'].fillna(0)

    required_cols = {'QuestionId', 'MC_Answer', 'StudentExplanation', 'target', 'is_correct'}
    missing = required_cols - set(df.columns)
    if missing:
        # 必要列が不足している場合は空マップを返す
        return {}

    # インデックス保持
    df = df.reset_index().rename(columns={'index': '_row_index'})

    # 代表例の選定
    examples_by_q = {}
    for qid, g in df.groupby('QuestionId'):
        # targetごとの頻度順に並べる
        counts = g['target'].value_counts().to_dict()
        # 各targetで代表例を選ぶ（説明文が最も長いものを優先）
        reps = []
        for tgt, _ in sorted(counts.items(), key=lambda x: -x[1]):
            sub = g[g['target'] == tgt]
            # StudentExplanation長が最大のものを選択
            best = sub.loc[sub['StudentExplanation'].astype(str).str.len().idxmax()]
            reps.append({
                'idx': int(best['_row_index']),
                'MC_Answer': best['MC_Answer'],
                'is_correct': int(best['is_correct']) == 1,
                'StudentExplanation': str(best['StudentExplanation']),
                'target': str(best['target'])
            })
            if len(reps) >= per_label * len(counts):
                # 念のため上限（per_label）でブレーク（counts長でper_label=1なら意味的に1/label）
                pass
        # 最大件数に制限
        examples_by_q[qid] = reps[:max_examples_per_q]

    return examples_by_q


def _format_fewshot_block(qid, fewshot_examples_by_qid, exclude_idx=None, k=0):
    if not fewshot_examples_by_qid or k <= 0:
        return ""
    exs = fewshot_examples_by_qid.get(qid, [])
    # 自身の行を除外
    if exclude_idx is not None:
        exs = [e for e in exs if e.get('idx') != exclude_idx]
    if not exs:
        return ""
    exs = exs[:k]
    lines = ["Examples (same QuestionId):"]
    for i, e in enumerate(exs, 1):
        status = "Yes" if e.get('is_correct') else "No"
        lines.extend([
            f"Example {i}:",
            f"Answer: {e.get('MC_Answer')}",
            f"Correct?: {status}",
            f"Explanation: {e.get('StudentExplanation')}",
            f"Label: {e.get('target')}",
            ""
        ])
    return "\n".join(lines) + "\n\n"


def format_input(row, label_names=None, fewshot_examples_by_qid=None, fewshot_k=0):
    """入力データをモデル用プロンプトにフォーマット
    - label_names: 出力可能なラベル列挙（例: 65クラス）
    - fewshot_examples_by_qid: {QuestionId: [few-shot例...]}
    - fewshot_k: few-shotの例数（0で無効）
    """
    status = "Yes" if row["is_correct"] else "No"

    label_instruction = _build_label_instruction(label_names)
    fewshot_block = _format_fewshot_block(
        qid=row['QuestionId'],
        fewshot_examples_by_qid=fewshot_examples_by_qid,
        exclude_idx=getattr(row, 'name', None),
        k=fewshot_k,
    )

    # 改良版プロンプト（ラベル列挙 + 同一QuestionIdのfew-shot例を先頭に提示）
    prompt = (
        "<|user|>\n"
        f"[Mathematical Misconception Analysis Task]\n\n"
        + label_instruction +
        fewshot_block +
        "Target case:\n"
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
