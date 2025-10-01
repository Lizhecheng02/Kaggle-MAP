"""
ヘルパー関数群

・PEFTの modules_to_save に分類ヘッドを含めるための自動検出
  submit.py では AutoModelForSequenceClassification をベースに LoRA アダプターを適用するため、
  学習時に分類ヘッド（例: `score` や `classifier`）を adapter 側に保存しておく必要がある。
"""

from typing import List


def detect_classifier_modules_to_save(model) -> List[str]:
    """
    モデル内の分類ヘッド名を推定し、PEFT の `modules_to_save` に渡すリストを返す。

    一般的な SequenceClassification モデルでは `score` または `classifier` が使用される。
    config.num_labels を持つ線形層であれば候補とみなす。
    """
    candidates = []
    possible_names = ["score", "classifier", "classification_head"]
    num_labels = getattr(getattr(model, "config", None), "num_labels", None)

    for name in possible_names:
        if hasattr(model, name):
            layer = getattr(model, name)
            try:
                out_features = getattr(layer, "out_features", None)
                if (num_labels is not None and out_features == num_labels) or out_features is None:
                    candidates.append(name)
            except Exception:
                # out_features が取れない場合でも候補として追加（後段で無視される可能性あり）
                candidates.append(name)

    # 重複除去
    return list(dict.fromkeys(candidates))

