import numpy as np
import pytest

from utils import compute_map3
from config import MAP_K, METRIC_NAME


def _manual_mapk(preds: np.ndarray, labels: np.ndarray, k: int) -> float:
    # 手計算のMAP@K（numpyのみ）
    x = preds - preds.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    probs = exp_x / exp_x.sum(axis=1, keepdims=True)
    topk = np.argpartition(-probs, kth=range(k), axis=1)[:, :k]
    row_idx = np.arange(probs.shape[0])[:, None]
    topk_sorted = topk[row_idx, np.argsort(-probs[row_idx, topk], axis=1)]

    s = 0.0
    for i, y in enumerate(labels):
        for r, p in enumerate(topk_sorted[i], start=1):
            if p == y:
                s += 1.0 / r
                break
    return s / len(labels)


def test_compute_map3_key_and_shape_tuple_input():
    # 5サンプル、4クラス
    logits = np.array([
        [10,  2,  1,  0],  # 正解=0 rank1
        [ 1, 10,  2,  0],  # 正解=1 rank1
        [ 1,  2, 10,  0],  # 正解=2 rank1
        [ 2,  1,  0, 10],  # 正解=3 rank1
        [ 2,  3,  4,  5],  # 正解=3 rank1
    ], dtype=float)
    labels = np.array([0, 1, 2, 3, 3])

    out = compute_map3((logits, labels))
    assert METRIC_NAME in out, "compute_metrics が設定されたキー名を返していません"

    expected = _manual_mapk(logits, labels, MAP_K)
    assert np.isclose(out[METRIC_NAME], expected, atol=1e-8)


def test_compute_map3_partial_ranks():
    # 正解が2位/3位に現れるケースを含める
    logits = np.array([
        [2, 3, 1, 0],   # 正解=0 -> rank2 (寄与=1/2)
        [2, 1, 3, 0],   # 正解=1 -> rank3 (寄与=1/3)
        [3, 2, 1, 0],   # 正解=2 -> rank3 (寄与=1/3)
        [5, 4, 3, 2],   # 正解=3 -> rank4 (寄与=0)
    ], dtype=float)
    labels = np.array([0, 1, 2, 3])

    out = compute_map3((logits, labels))
    expected = _manual_mapk(logits, labels, MAP_K)
    assert np.isclose(out[METRIC_NAME], expected, atol=1e-8)

