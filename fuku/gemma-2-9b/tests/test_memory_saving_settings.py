from pathlib import Path


def test_no_output_hidden_states_request():
    base = Path(__file__).resolve().parents[1]
    train_path = base / "train.py"
    assert train_path.exists(), "train.py が見つかりません"
    src = train_path.read_text(encoding="utf-8")
    assert "output_hidden_states=True" not in src, "hidden_states の全層保持を強制するコードは削除してください"


def test_use_cache_disabled_in_model_config():
    base = Path(__file__).resolve().parents[1]
    train_path = base / "train.py"
    src = train_path.read_text(encoding="utf-8")
    assert "use_cache = False" in src, "学習時に use_cache を無効化してメモリを節約してください"


def test_gradient_checkpointing_flag_from_config():
    base = Path(__file__).resolve().parents[1]
    train_path = base / "train.py"
    src = train_path.read_text(encoding="utf-8")
    assert (
        "gradient_checkpointing=USE_GRADIENT_CHECKPOINTING" in src
    ), "TrainingArguments は設定ファイルの USE_GRADIENT_CHECKPOINTING を参照してください"

