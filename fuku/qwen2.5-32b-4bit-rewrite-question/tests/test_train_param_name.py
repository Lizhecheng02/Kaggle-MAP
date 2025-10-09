from pathlib import Path


def test_train_uses_evaluation_strategy_not_eval_strategy():
    p = Path("fuku/qwen2.5-32b-4bit/train.py")
    text = p.read_text(encoding="utf-8")
    assert "evaluation_strategy" in text
    assert "eval_strategy" not in text

