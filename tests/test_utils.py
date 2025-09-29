import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

# テスト用に対象モジュールフォルダを import path に追加
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fuku', 'phi-4quack'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import (
    prepare_correct_answers,
    format_input,
    tokenize_dataset,
    create_submission,
)


def test_prepare_correct_answers_basic():
    df = pd.DataFrame({
        'Category': ['True_A', 'False_A', 'True_B', 'False_B', 'True_A'],
        'QuestionId': [1, 1, 2, 2, 1],
        'MC_Answer': ['X', 'Y', 'Z', 'Z', 'X'],
    })
    correct = prepare_correct_answers(df)
    # 正解候補は QuestionId ごとに1行、is_correct=1
    assert 'is_correct' in correct.columns
    assert set(correct['QuestionId']) == {1, 2}
    assert (correct['is_correct'] == 1).all()


def test_format_input_yes_no():
    row = pd.Series({
        'is_correct': 1,
        'QuestionText': 'What is 2+2?',
        'MC_Answer': '4',
        'StudentExplanation': 'Because 2 and 2 make 4.'
    })
    prompt = format_input(row)
    assert 'Correct?: Yes' in prompt

    row['is_correct'] = 0
    prompt = format_input(row)
    assert 'Correct?: No' in prompt


def test_create_submission_top3():
    # 2サンプル, ロジット3クラス
    logits = np.array([[3.0, 1.0, 2.0], [0.1, 0.2, 0.3]], dtype=np.float32)

    class Predictions:
        def __init__(self, preds):
            self.predictions = preds

    preds = Predictions(logits)
    le = LabelEncoder()
    le.fit(['A', 'B', 'C'])
    test_df = pd.DataFrame({'row_id': [0, 1]})

    sub = create_submission(preds, test_df, le)
    assert list(sub.columns) == ['row_id', 'Category:Misconception']
    assert len(sub) == 2
    assert len(sub.loc[0, 'Category:Misconception'].split()) == 3


def test_tokenize_dataset_with_dummy_tokenizer():
    # ダミートークナイザ: 空白で分割してID化
    class DummyTokenizer:
        pad_token_id = 0

        def __call__(self, texts, padding=False, truncation=True, max_length=10, return_tensors=None):
            if isinstance(texts, str):
                texts = [texts]
            input_ids = []
            attn = []
            for t in texts:
                ids = list(range(1, min(max_length, len(t.split())) + 1))
                input_ids.append(ids)
                attn.append([1] * len(ids))
            return {'input_ids': input_ids, 'attention_mask': attn}

    df = pd.DataFrame({'text': ['a b c', 'd e'], 'label': [0, 1]})
    ds = Dataset.from_pandas(df)
    tokenizer = DummyTokenizer()
    tokenized = tokenize_dataset(ds, tokenizer, max_len=10)

    batch = next(iter(tokenized))
    assert 'input_ids' in batch and 'attention_mask' in batch and 'label' in batch
