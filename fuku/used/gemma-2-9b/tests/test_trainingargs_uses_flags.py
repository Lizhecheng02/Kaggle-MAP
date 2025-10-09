from pathlib import Path


def test_trainingargs_uses_bf16_flags():
    base = Path(__file__).resolve().parents[1]
    train_path = base / "train.py"
    assert train_path.exists(), "train.py が見つかりません"

    src = train_path.read_text(encoding="utf-8")
    # TrainingArguments 内で設定フラグを使っていることを確認
    assert "bf16=USE_BF16" in src, "TrainingArguments で bf16=USE_BF16 が設定されていません"
    assert "fp16=USE_FP16" in src, "TrainingArguments で fp16=USE_FP16 が設定されていません"

