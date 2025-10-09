import os
from pathlib import Path


def test_save_best_map3_callback_saves_only_on_improvement(tmp_path):
    # import は実体関数使用のために行う
    from train import SaveBestMap3Callback

    # ダミーモデル/トークナイザー
    saved_paths = []

    class DummyModel:
        def save_pretrained(self, path):
            saved_paths.append(Path(path))

    class DummyTokenizer:
        def save_pretrained(self, path):
            saved_paths.append(Path(path))

    cb = SaveBestMap3Callback(save_dir=str(tmp_path), tokenizer=DummyTokenizer())

    # 1回目: 改善あり
    cb.on_evaluate(args=None, state=None, control=None, metrics={"eval_map@3": 0.5}, model=DummyModel())
    assert any(p.name == "best_map3" for p in saved_paths), "best_map3 への保存が行われていません"

    saved_count_after_first = len(saved_paths)

    # 2回目: 改善なし -> 保存されない
    cb.on_evaluate(args=None, state=None, control=None, metrics={"eval_map@3": 0.4}, model=DummyModel())
    assert len(saved_paths) == saved_count_after_first, "改善なしで保存が行われています"

