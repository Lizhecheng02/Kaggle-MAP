from pathlib import Path


def test_submit_uses_precision_flags():
    base = Path(__file__).resolve().parents[1]
    submit_path = base / "submit.py"
    assert submit_path.exists(), "submit.py が見つかりません"

    src = submit_path.read_text(encoding="utf-8")

    # USE_BF16/USE_FP16 の両フラグを参照していること
    assert "USE_BF16" in src, "submit.py が USE_BF16 を参照していません"
    assert "USE_FP16" in src, "submit.py が USE_FP16 を参照していません"

    # dtype 変数を torch_dtype に適用していること
    assert "torch_dtype=dtype" in src, "from_pretrained で torch_dtype=dtype が設定されていません"

    # dtype の選択ロジックが存在すること
    assert "dtype = (" in src and "USE_BF16" in src and "USE_FP16" in src,
    "dtype 選択ロジックが設定フラグを用いていません"

