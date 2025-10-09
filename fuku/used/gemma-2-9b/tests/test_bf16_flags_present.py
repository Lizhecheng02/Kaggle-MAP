from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path


def test_config_has_bf16_flags():
    base = Path(__file__).resolve().parents[1]
    cfg_path = base / "config_sample.py"
    assert cfg_path.exists(), "config_sample.py が見つかりません"

    spec = spec_from_file_location("cfg", str(cfg_path))
    mod = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    assert hasattr(mod, "USE_BF16"), "USE_BF16 が定義されていません"
    assert hasattr(mod, "USE_FP16"), "USE_FP16 が定義されていません"
    assert isinstance(mod.USE_BF16, bool)
    assert isinstance(mod.USE_FP16, bool)
    # 既定では bf16 を有効化
    assert mod.USE_BF16 is True
    assert mod.USE_FP16 is False

