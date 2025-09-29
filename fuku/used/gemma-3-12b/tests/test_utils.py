import numpy as np
import torch

from utils import format_input, compute_map3


def test_format_input_contains_fields():
    row = {
        'QuestionText': 'What is 2+2?',
        'MC_Answer': '4',
        'StudentExplanation': 'I added two and two.',
        'is_correct': 1,
    }
    prompt = format_input(row)
    assert '[Mathematical Misconception Analysis Task]' in prompt
    assert 'Question: What is 2+2?' in prompt
    assert 'Answer: 4' in prompt
    assert 'Correct?: Yes' in prompt
    assert 'Explanation: I added two and two.' in prompt


def test_compute_map3_perfect():
    # 3クラス, バッチ2。正解が top-1/2/3 に入る動作の単体チェック
    # 完全一致のケースを作る
    logits = np.array([
        [10.0, 1.0, 0.0],    # 正解クラス0
        [0.1, 0.2, 5.0],     # 正解クラス2
        [0.2, 4.0, 0.3],     # 正解クラス1
    ], dtype=np.float32)
    labels = np.array([0, 2, 1], dtype=np.int64)

    # transformers.Trainer の compute_metrics 互換: (logits, labels)
    result = compute_map3((logits, labels))
    assert 'map@3' in result
    # 3サンプルともtop3に正解が入っているので1.0
    assert result['map@3'] == 1.0

