import sys
import types
import importlib
import pytest


def test_import_submit_without_peft_raises_module_not_found(monkeypatch):
    # ダミー transformers モジュール（トップレベルimportに必要なシンボルのみ）
    dummy_tf = types.ModuleType("transformers")
    dummy_tf.AutoModelForSequenceClassification = object
    dummy_tf.AutoTokenizer = object
    dummy_tf.Trainer = object
    dummy_tf.TrainingArguments = object
    dummy_tf.DataCollatorWithPadding = object
    monkeypatch.setitem(sys.modules, "transformers", dummy_tf)

    # ダミー datasets モジュール（トップレベルimportに必要なシンボルのみ）
    dummy_ds = types.ModuleType("datasets")
    class _Dataset:
        pass
    dummy_ds.Dataset = _Dataset
    monkeypatch.setitem(sys.modules, "datasets", dummy_ds)

    # torch が未インストール環境でもトップレベルimportが通るようにスタブ
    dummy_torch = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    # peft を明示的に未登録にして import 時に例外が発生することを確認
    monkeypatch.setitem(sys.modules, "peft", None)
    sys.modules.pop("peft", None)

    # 既にキャッシュされている submit をクリア
    sys.modules.pop("submit", None)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("submit")

    # 他テストへの副作用を避けるため submit を確実に除去
    sys.modules.pop("submit", None)

