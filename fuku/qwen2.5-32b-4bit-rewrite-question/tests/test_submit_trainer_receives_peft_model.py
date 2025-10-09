import sys
import types
import importlib
import numpy as np
import pandas as pd


def test_trainer_receives_peft_model_not_merged(monkeypatch, tmp_path):
    # Dummy PEFT with merge_and_unload defined（呼ばれないことを期待）
    class DummyBaseModel:
        class Cfg:
            pad_token_id = None

        def __init__(self):
            self.config = DummyBaseModel.Cfg()
            class Inner:
                def __init__(self):
                    class Cfg2:
                        pad_token_id = None
                    self.config = Cfg2()
            self.model = Inner()

    class DummyPeftModel:
        def __init__(self, base):
            self.base_model = base

        @classmethod
        def from_pretrained(cls, model, path):
            return cls(model)

        def eval(self):
            return None

        def merge_and_unload(self):
            # もし呼ばれたら純ベースモデルを返す（本テストでは呼ばれない想定）
            return self.base_model

    dummy_peft = types.ModuleType("peft")
    dummy_peft.PeftModel = DummyPeftModel
    dummy_peft.PeftConfig = object
    monkeypatch.setitem(sys.modules, "peft", dummy_peft)

    # transformers のスタブ
    constructed_quant_configs = {}

    class DummyBitsAndBytesConfig:
        def __init__(self, **kwargs):
            constructed_quant_configs.update(kwargs)

    class DummyAutoModelForSeqCls:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs.get("quantization_config") is not None
            return DummyBaseModel()

    class DummyTokenizer:
        def __init__(self):
            self.pad_token = None
            self.pad_token_id = None
            self.eos_token = "<eos>"
            self.eos_token_id = 0

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class DummyTrainingArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyTrainer:
        def __init__(self, model=None, processing_class=None, data_collator=None, args=None):
            # ここでPEFTラッパーのまま渡ってきていることを検証
            assert model.__class__.__name__ == "DummyPeftModel"
            self._n = 2

        class _Pred:
            def __init__(self, n):
                self.predictions = np.zeros((n, 3), dtype=np.float32)

        def predict(self, dataset):
            n = getattr(dataset, "__len__", lambda: 2)()
            return DummyTrainer._Pred(n)

    import transformers as _tf
    monkeypatch.setattr(_tf, "BitsAndBytesConfig", DummyBitsAndBytesConfig, raising=True)
    monkeypatch.setattr(_tf, "AutoModelForSequenceClassification", DummyAutoModelForSeqCls, raising=True)
    monkeypatch.setattr(_tf, "AutoTokenizer", DummyTokenizer, raising=True)
    monkeypatch.setattr(_tf, "Trainer", DummyTrainer, raising=True)
    monkeypatch.setattr(_tf, "TrainingArguments", DummyTrainingArguments, raising=True)

    # Dataset.from_pandas の簡易スタブ
    import datasets as _ds

    class DummyDataset:
        def __init__(self, df):
            self._len = len(df)

        def __len__(self):
            return self._len

    monkeypatch.setattr(_ds.Dataset, "from_pandas", staticmethod(lambda df: DummyDataset(df)), raising=True)

    # IO 関連をスタブ
    def fake_read_csv(path):
        if "test" in str(path):
            return pd.DataFrame([
                {"row_id": "r1", "QuestionId": 1, "MC_Answer": "A", "QuestionText": "Q1", "StudentExplanation": "E1"},
                {"row_id": "r2", "QuestionId": 2, "MC_Answer": "B", "QuestionText": "Q2", "StudentExplanation": "E2"},
            ])
        else:
            return pd.DataFrame([
                {"Category": "True_x", "Misconception": "m1", "QuestionId": 1, "MC_Answer": "A", "QuestionText": "Q1", "StudentExplanation": "E1"},
                {"Category": "False_x", "Misconception": "m2", "QuestionId": 2, "MC_Answer": "B", "QuestionText": "Q2", "StudentExplanation": "E2"},
            ])

    import utils as _utils

    def fake_prepare_correct_answers(train_df):
        return pd.DataFrame({"QuestionId": [1, 2], "MC_Answer": ["A", "B"], "is_correct": [1, 0]})

    def fake_tokenize_dataset(ds, tokenizer, max_len):
        return ds

    def fake_create_submission(predictions, test_df, le):
        return pd.DataFrame({"row_id": test_df["row_id"].values, "Category:Misconception": ["A:X B:Y C:Z"] * len(test_df)})

    monkeypatch.setattr(pd, "read_csv", fake_read_csv, raising=True)
    monkeypatch.setattr(_utils, "prepare_correct_answers", fake_prepare_correct_answers, raising=True)
    monkeypatch.setattr(_utils, "tokenize_dataset", fake_tokenize_dataset, raising=True)
    monkeypatch.setattr(_utils, "create_submission", fake_create_submission, raising=True)

    # LabelEncoder ロードをスタブ
    import joblib as _joblib
    class DummyLE:
        classes_ = np.array(["A:X", "B:Y", "C:Z"])
        def inverse_transform(self, idx):
            return self.classes_[idx]
    monkeypatch.setattr(_joblib, "load", lambda p: DummyLE(), raising=True)

    # 実行
    monkeypatch.chdir(tmp_path)
    import submit
    submit.main()

