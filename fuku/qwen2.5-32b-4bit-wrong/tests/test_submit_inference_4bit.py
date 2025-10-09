import sys
import types
import importlib
import numpy as np
import pandas as pd
import torch


def test_submit_uses_trained_lora_and_4bit(monkeypatch, tmp_path):
    # ダミー peft モジュールを注入（PeftModel.from_pretrained が呼ばれるかを検証）
    class DummyPeftModel:
        last_loaded_path = None

        def __init__(self, base_model):
            self.base_model = base_model

        @classmethod
        def from_pretrained(cls, model, path):
            cls.last_loaded_path = path
            return cls(model)

        def eval(self):
            return None

    dummy_peft = types.ModuleType("peft")
    dummy_peft.PeftModel = DummyPeftModel
    dummy_peft.PeftConfig = object
    sys.modules["peft"] = dummy_peft

    # transformers をパッチ
    constructed_quant_configs = {}

    class DummyBitsAndBytesConfig:
        def __init__(self, **kwargs):
            constructed_quant_configs.update(kwargs)

    class DummyBaseModel:
        def __init__(self):
            class Cfg:
                pad_token_id = None
            self.config = Cfg()
            # 内部 model を持つケースにも対応
            class Inner:
                def __init__(self):
                    class Cfg2:
                        pad_token_id = None
                    self.config = Cfg2()
            self.model = Inner()

    class DummyAutoModelForSeqCls:
        last_kwargs = None

        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            cls.last_kwargs = kwargs
            # 量子化設定の存在を検証
            qc = kwargs.get("quantization_config")
            assert qc is not None, "quantization_config が渡されていません"
            # DummyBitsAndBytesConfig により kwargs が constructed_quant_configs に保存される
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

    class DummyTrainer:
        def __init__(self, model=None, processing_class=None, data_collator=None, args=None):
            self.model = model
            self.processing_class = processing_class
            self.data_collator = data_collator
            self.args = args

        class _Pred:
            def __init__(self, arr):
                self.predictions = arr

        def predict(self, dataset):
            # 入力データ数に応じてダミー予測を返す
            n = getattr(dataset, "__len__", lambda: 2)()
            # ラベル数は3に固定（ダミーのLabelEncoderに合わせる）
            arr = np.zeros((n, 3), dtype=np.float32)
            return DummyTrainer._Pred(arr)

    class DummyTrainingArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    # datasets.Dataset.from_pandas を簡略化
    class DummyDataset:
        def __init__(self, df):
            self._len = len(df)

        def __len__(self):
            return self._len

    def dummy_from_pandas(df):
        return DummyDataset(df)

    # pandas.read_csv をパッチ（必要なカラムのみの最小データを返す）
    def fake_read_csv(path):
        # train/test を判定
        if "test" in str(path):
            return pd.DataFrame([
                {"row_id": "r1", "QuestionId": 1, "MC_Answer": "A", "QuestionText": "Q1", "StudentExplanation": "E1"},
                {"row_id": "r2", "QuestionId": 2, "MC_Answer": "B", "QuestionText": "Q2", "StudentExplanation": "E2"},
            ])
        else:  # train
            return pd.DataFrame([
                {"Category": "True_x", "Misconception": "m1", "QuestionId": 1, "MC_Answer": "A", "QuestionText": "Q1", "StudentExplanation": "E1"},
                {"Category": "False_x", "Misconception": "m2", "QuestionId": 2, "MC_Answer": "B", "QuestionText": "Q2", "StudentExplanation": "E2"},
            ])

    # utils をパッチ
    import utils as _utils

    def fake_prepare_correct_answers(train_df):
        return pd.DataFrame({"QuestionId": [1, 2], "MC_Answer": ["A", "B"], "is_correct": [1, 0]})

    def fake_tokenize_dataset(ds, tokenizer, max_len):
        return ds  # 何もしない

    class DummyLE:
        def __init__(self):
            self.classes_ = np.array(["A:X", "B:Y", "C:Z"])  # 3クラス

        def inverse_transform(self, idx):
            return self.classes_[idx]

    def fake_joblib_load(path):
        return DummyLE()

    def fake_create_submission(predictions, test_df, le):
        # 形だけの提出DataFrameを返す
        return pd.DataFrame({
            "row_id": test_df["row_id"].values,
            "Category:Misconception": ["A:X B:Y C:Z"] * len(test_df),
        })

    monkeypatch.setitem(sys.modules, "transformers", importlib.import_module("transformers"))
    import transformers as _tf
    monkeypatch.setattr(_tf, "BitsAndBytesConfig", DummyBitsAndBytesConfig, raising=True)
    monkeypatch.setattr(_tf, "AutoModelForSequenceClassification", DummyAutoModelForSeqCls, raising=True)
    monkeypatch.setattr(_tf, "AutoTokenizer", DummyTokenizer, raising=True)
    monkeypatch.setattr(_tf, "Trainer", DummyTrainer, raising=True)
    monkeypatch.setattr(_tf, "TrainingArguments", DummyTrainingArguments, raising=True)

    import datasets as _ds
    monkeypatch.setattr(_ds.Dataset, "from_pandas", staticmethod(dummy_from_pandas), raising=True)

    monkeypatch.setattr(pd, "read_csv", fake_read_csv, raising=True)
    monkeypatch.setattr(_utils, "prepare_correct_answers", fake_prepare_correct_answers, raising=True)
    monkeypatch.setattr(_utils, "tokenize_dataset", fake_tokenize_dataset, raising=True)
    monkeypatch.setattr(_utils, "create_submission", fake_create_submission, raising=True)

    import joblib as _joblib
    monkeypatch.setattr(_joblib, "load", fake_joblib_load, raising=True)

    # submit を遅延インポート（上記パッチを有効にした状態で）
    import submit
    from config import BEST_MODEL_PATH, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE, BNB_4BIT_QUANT_STORAGE_DTYPE

    # SUBMISSION_OUTPUT_PATH がカレントに出力されるのを避けるため、cwd を一時変更
    monkeypatch.chdir(tmp_path)

    # 実行
    submit.main()

    # 4bit量子化の設定が期待通りか検証
    assert constructed_quant_configs.get("load_in_4bit") is True
    assert constructed_quant_configs.get("bnb_4bit_quant_type") == BNB_4BIT_QUANT_TYPE
    # dtype は torch の実体と比較
    assert constructed_quant_configs.get("bnb_4bit_compute_dtype") == getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    assert constructed_quant_configs.get("bnb_4bit_quant_storage_dtype") == getattr(torch, BNB_4BIT_QUANT_STORAGE_DTYPE)

    # 学習済みLoRAアダプタが読み込まれているか検証
    assert DummyPeftModel.last_loaded_path == BEST_MODEL_PATH

