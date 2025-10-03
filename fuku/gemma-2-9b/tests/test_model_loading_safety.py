from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import types


def _import_train_module():
    base = Path(__file__).resolve().parents[1]
    train_path = base / "train.py"
    assert train_path.exists(), "train.py が見つかりません"
    spec = spec_from_file_location("train_mod", str(train_path))
    mod = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_enable_input_require_grads_called_when_gc_flag():
    # コードに enable_input_require_grads 呼び出しを含むことを保証
    base = Path(__file__).resolve().parents[1]
    src = (base / "train.py").read_text(encoding="utf-8")
    assert "enable_input_require_grads()" in src,
    "勾配チェックポイント時に enable_input_require_grads() を呼び出してください"


def test_awq_detection_heuristic():
    train_mod = _import_train_module()
    assert hasattr(train_mod, "_detect_awq_modules"), "_detect_awq_modules が見つかりません"

    # ダミーの AWQ 風レイヤを持つモデルで True を返すこと
    import torch.nn as nn

    class WQLinear_Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(nn.init.xavier_uniform_(nn.Parameter(nn.Tensor(4, 4)).data))
        def forward(self, x):
            return x

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = WQLinear_Dummy()

    m = DummyModel()
    assert train_mod._detect_awq_modules(m) is True, "AWQ 風クラス名を検知できていません"

    # ふつうのモデルでは False
    class PlainModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

    pm = PlainModel()
    assert train_mod._detect_awq_modules(pm) is False, "非AWQモデルを誤検出しています"

