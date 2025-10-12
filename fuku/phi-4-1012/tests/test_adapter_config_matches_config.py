import json
import os
import importlib
import pytest


def _load_adapter_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_lora_adapter_config_matches_config_constants():
    cfg = importlib.import_module("config")
    adapter_path = os.path.join(cfg.BEST_MODEL_PATH, "adapter_config.json")

    if not os.path.exists(adapter_path):
        pytest.skip(f"adapter_config.json not found: {adapter_path}")

    data = _load_adapter_config(adapter_path)

    assert data.get("peft_type") == "LORA"
    assert data.get("task_type") == "SEQ_CLS"

    assert data.get("r") == cfg.LORA_RANK
    assert data.get("lora_alpha") == cfg.LORA_ALPHA
    assert pytest.approx(float(data.get("lora_dropout", 0.0))) == float(cfg.LORA_DROPOUT)

    tm_saved = set(data.get("target_modules") or [])
    tm_cfg = set(cfg.LORA_TARGET_MODULES)
    assert tm_saved == tm_cfg, f"target_modules mismatch: saved={tm_saved} cfg={tm_cfg}"

    # bias表現の整合（adapter_configには "bias" と "lora_bias" の両方が入ることがある）
    bias_saved = data.get("bias")
    lora_bias_flag = data.get("lora_bias")
    if cfg.LORA_BIAS == "none":
        assert bias_saved == "none"
        # ライブラリ実装によりbool/strの差が出ることがあるため緩く見る
        assert (lora_bias_flag is False) or (lora_bias_flag == "none")
    else:
        # それ以外のケースは最低限、adapter_configに何らかの形で保存されていることを確認
        assert bias_saved is not None

