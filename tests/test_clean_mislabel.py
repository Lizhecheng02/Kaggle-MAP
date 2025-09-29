import os
import sys
import pandas as pd

# テスト用に対象モジュールフォルダを import path に追加
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fuku', 'phi-4quack'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import clean_mislabel_entries, compute_split_key
from config import MISLABEL_QID, MISLABEL_CORRECT_ANSWER


def _make_df_for_mislabel():
    # 3行: 2つが修正対象、1つは対象外
    return pd.DataFrame({
        'QuestionId': [MISLABEL_QID, MISLABEL_QID, 99999],
        'Category': ['False_suffix', 'True_suffix', 'True_suffix'],
        'MC_Answer': [MISLABEL_CORRECT_ANSWER, 'wrong', 'anything'],
        'QuestionText': ['Q', 'Q', 'Q'],
        'StudentExplanation': ['E', 'E', 'E']
    })


def test_clean_mislabel_fix():
    df = _make_df_for_mislabel()
    fixed = clean_mislabel_entries(df, mode='fix')
    # 1行目はTrue_に、2行目はFalse_に修正される
    assert fixed.loc[0, 'Category'].startswith('True_')
    assert fixed.loc[1, 'Category'].startswith('False_')
    # 対象外は変更なし
    assert fixed.loc[2, 'Category'] == 'True_suffix'


def test_clean_mislabel_remove():
    df = _make_df_for_mislabel()
    removed = clean_mislabel_entries(df, mode='remove')
    # 修正対象2行が削除され、1行残る
    assert len(removed) == 1
    assert removed.iloc[0]['QuestionId'] != MISLABEL_QID


def test_clean_mislabel_ignore():
    df = _make_df_for_mislabel()
    ignored = clean_mislabel_entries(df, mode='ignore')
    # 無視では何も変わらない
    assert ignored.equals(df)


def test_compute_split_key():
    df = _make_df_for_mislabel()
    # ラベル列が必要
    df = df.assign(label=[0, 1, 0])
    codes = compute_split_key(df)
    assert len(codes) == len(df)
    assert codes.dtype.name.startswith('int')
