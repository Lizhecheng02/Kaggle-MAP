import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import process_targets
from config import (
    TARGET_QUESTION_ID_COL,
    TARGET_BASECATEGORY_COL,
    TARGET_MISCONCEPTION_COL,
    TARGET_NEITHER_KEYWORD,
    TARGET_NEITHER_PREFIX,
)


def test_process_targets_neither_case():
    row = pd.Series({
        TARGET_QUESTION_ID_COL: 1234,
        TARGET_BASECATEGORY_COL: f"{TARGET_NEITHER_KEYWORD}_Something",
        TARGET_MISCONCEPTION_COL: "SomeMisc",
    })
    got = process_targets(row)
    assert got == f"{TARGET_NEITHER_PREFIX}:{row[TARGET_BASECATEGORY_COL]}:{row[TARGET_MISCONCEPTION_COL]}"


def test_process_targets_normal_case():
    row = pd.Series({
        TARGET_QUESTION_ID_COL: 5678,
        TARGET_BASECATEGORY_COL: "Algebra",
        TARGET_MISCONCEPTION_COL: "MisA",
    })
    got = process_targets(row)
    assert got == f"{row[TARGET_QUESTION_ID_COL]}:{row[TARGET_BASECATEGORY_COL]}:{row[TARGET_MISCONCEPTION_COL]}"


def test_label_encoding_after_processing():
    df = pd.DataFrame([
        {TARGET_QUESTION_ID_COL: 1, TARGET_BASECATEGORY_COL: f"{TARGET_NEITHER_KEYWORD}_Cat", TARGET_MISCONCEPTION_COL: "M1"},
        {TARGET_QUESTION_ID_COL: 1, TARGET_BASECATEGORY_COL: "Geo", TARGET_MISCONCEPTION_COL: "M2"},
        {TARGET_QUESTION_ID_COL: 2, TARGET_BASECATEGORY_COL: "Geo", TARGET_MISCONCEPTION_COL: "M2"},
    ])
    df["target"] = df.apply(process_targets, axis=1)

    le = LabelEncoder()
    labels = le.fit_transform(df["target"])

    # クラス数はユニークなターゲットと一致
    assert len(le.classes_) == df["target"].nunique()
    # 逆変換で元のターゲットに戻る
    inv = le.inverse_transform(labels)
    assert list(inv) == list(df["target"])  # 学習直後は同順


def test_process_targets_category_fallback():
    # BaseCategoryが存在しない場合でもCategoryでフォールバックできること
    row = pd.Series({
        TARGET_QUESTION_ID_COL: 42,
        # TARGET_BASECATEGORY_COL はあえて欠落
        "Category": "Algebra",
        TARGET_MISCONCEPTION_COL: "MZ",
    })
    got = process_targets(row)
    assert got == f"{row[TARGET_QUESTION_ID_COL]}:{row['Category']}:{row[TARGET_MISCONCEPTION_COL]}"
