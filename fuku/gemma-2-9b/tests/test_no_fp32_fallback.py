from pathlib import Path


def test_no_torch_float32_in_code():
    base = Path(__file__).resolve().parents[1]

    for fname in ("train.py", "submit.py"):
        path = base / fname
        assert path.exists(), f"{fname} が見つかりません"
        src = path.read_text(encoding="utf-8")
        assert "torch.float32" not in src, f"{fname} に torch.float32 のフォールバックが残っています"

