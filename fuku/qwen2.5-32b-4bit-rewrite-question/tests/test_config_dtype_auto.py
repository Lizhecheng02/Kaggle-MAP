import sys
import types
import importlib


def test_bnb_4bit_compute_dtype_auto_selects_fp16_when_bf16_unsupported(monkeypatch):
    # torch のスタブ（GPU 利用可能だが bf16 は非対応）
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # 既存の config をクリアして再インポート
    sys.modules.pop("config", None)
    import config as cfg

    assert cfg.BNB_4BIT_COMPUTE_DTYPE == "float16"


def test_bnb_4bit_compute_dtype_uses_bf16_when_supported(monkeypatch):
    # torch のスタブ（GPU 利用可能で bf16 も対応）
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # 既存の config をクリアして再インポート
    sys.modules.pop("config", None)
    import config as cfg

    assert cfg.BNB_4BIT_COMPUTE_DTYPE == "bfloat16"

