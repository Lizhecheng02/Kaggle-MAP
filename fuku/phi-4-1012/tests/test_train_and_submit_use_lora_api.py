import os
import re
import pytest


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def test_train_uses_lora_api():
    # train.py は本ディレクトリ外（../phi-4/train.py）にあるケースを考慮
    candidates = [
        os.path.join(".", "train.py"),
        os.path.join("..", "phi-4", "train.py"),
    ]
    train_path = next((p for p in candidates if os.path.exists(p)), None)
    if train_path is None:
        pytest.skip("train.py not found in expected locations")

    src = _read(train_path)
    assert "LoraConfig(" in src, "LoraConfig is not constructed in train.py"
    assert "get_peft_model(" in src, "get_peft_model is not called in train.py"

    # target_modules 引数が設定されていること
    assert re.search(r"target_modules\s*=\s*LORA_TARGET_MODULES", src) is not None


def test_submit_loads_peft_adapter():
    candidates = [
        os.path.join(".", "submit.py"),
        os.path.join("..", "phi-4", "submit.py"),
    ]
    submit_path = next((p for p in candidates if os.path.exists(p)), None)
    if submit_path is None:
        pytest.skip("submit.py not found in expected locations")

    src = _read(submit_path)
    assert "from peft import PeftModel" in src or "import peft" in src
    assert "PeftModel.from_pretrained(" in src, "submit.py does not load LoRA adapter via PeftModel.from_pretrained"

