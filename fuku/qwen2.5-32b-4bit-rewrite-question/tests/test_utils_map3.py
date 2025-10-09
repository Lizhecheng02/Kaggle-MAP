import numpy as np
from utils import compute_map3


def test_compute_map3_basic():
    # 3クラス分類の例
    logits = np.array([
        [3.0, 1.0, 0.0],  # 正解0: top1 → 1.0
        [1.0, 2.0, 3.0],  # 正解1: 順位2 → 0.5
        [3.0, 2.0, 1.0],  # 正解2: 順位3 → 1/3
    ], dtype=np.float32)
    labels = np.array([0, 1, 2], dtype=np.int64)

    result = compute_map3((logits, labels))
    expected = (1.0 + 0.5 + (1.0 / 3.0)) / 3.0
    assert "map@3" in result
    assert abs(result["map@3"] - expected) < 1e-6

