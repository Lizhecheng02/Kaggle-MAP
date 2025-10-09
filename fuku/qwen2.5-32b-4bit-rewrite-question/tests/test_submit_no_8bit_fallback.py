import sys
import types
import importlib
import pytest
import numpy as np


def test_submit_does_not_fallback_to_8bit_on_4bit_failure(monkeypatch, tmp_path):
    # ---- ダミー peft （インポート通過のみ、呼ばれないことを検証） ----
    class _DummyPeftModel:
        called = False

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.called = True
            raise AssertionError("PEFT should not be loaded when 4bit base load fails")

    dummy_peft = types.ModuleType("peft")
    dummy_peft.PeftModel = _DummyPeftModel
    dummy_peft.PeftConfig = object
    monkeypatch.setitem(sys.modules, "peft", dummy_peft)

    # ---- ダミー transformers （最初の from_pretrained 呼び出しで 4bit 失敗を模擬） ----
    constructed_quant_configs = {}

    class DummyBitsAndBytesConfig:
        def __init__(self, **kwargs):
            constructed_quant_configs.update(kwargs)

    class DummyAutoModelForSeqCls:
        call_count = 0

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.call_count += 1
            # 4bitロードエラーを模擬（元コードが検知していた文言を含める）
            raise ValueError("Some modules are dispatched on the CPU or the disk")

    class DummyTokenizer:
        pass

    class DummyTrainer:
        pass

    class DummyTrainingArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyDataCollatorWithPadding:
        def __init__(self, *args, **kwargs):
            pass

    dummy_tf = types.ModuleType("transformers")
    dummy_tf.BitsAndBytesConfig = DummyBitsAndBytesConfig
    dummy_tf.AutoModelForSequenceClassification = DummyAutoModelForSeqCls
    dummy_tf.AutoTokenizer = DummyTokenizer
    dummy_tf.Trainer = DummyTrainer
    dummy_tf.TrainingArguments = DummyTrainingArguments
    dummy_tf.DataCollatorWithPadding = DummyDataCollatorWithPadding
    monkeypatch.setitem(sys.modules, "transformers", dummy_tf)

    # ---- ダミー datasets ----
    dummy_ds = types.ModuleType("datasets")
    class _Dataset:
        @staticmethod
        def from_pandas(df):
            class _D:
                def __len__(self):
                    return len(getattr(df, "index", []))
            return _D()
    dummy_ds.Dataset = _Dataset
    monkeypatch.setitem(sys.modules, "datasets", dummy_ds)

    # ---- joblib.load をスタブ（LabelEncoder読込を回避） ----
    import joblib as _joblib
    class _DummyLE:
        classes_ = np.array(["A:X", "B:Y", "C:Z"])
        def inverse_transform(self, idx):
            return self.classes_[idx]
    monkeypatch.setattr(_joblib, "load", lambda p: _DummyLE(), raising=True)

    # ---- 実行 ----
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("submit", None)
    import submit

    with pytest.raises(RuntimeError):
        submit.main()

    # 8bit への再呼び出しは行われず、最初の1回のみであること
    assert DummyAutoModelForSeqCls.call_count == 1
    # 4bit失敗のため LoRA のロードは呼ばれていないこと
    assert _DummyPeftModel.called is False

