import os
import io
import tempfile
from train import SaveBestMap3Callback


class _DummyModel:
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "adapter.bin"), "wb") as f:
            f.write(b"dummy")


class _DummyTokenizer:
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer.json"), "w") as f:
            f.write("{}")


def test_save_best_map3_callback_saves_on_improvement():
    with tempfile.TemporaryDirectory() as tmp:
        cb = SaveBestMap3Callback(save_dir=tmp, tokenizer=_DummyTokenizer())

        control = type("Control", (), {})()
        model = _DummyModel()

        # 1回目（改善）
        cb.on_evaluate(args=None, state=None, control=control, metrics={"eval_map@3": 0.70}, model=model)
        assert abs(cb.best_map3 - 0.70) < 1e-9
        assert os.path.exists(os.path.join(tmp, "best_map3", "adapter.bin"))

        # 2回目（非改善）
        cb.on_evaluate(args=None, state=None, control=control, metrics={"eval_map@3": 0.65}, model=model)
        assert abs(cb.best_map3 - 0.70) < 1e-9

