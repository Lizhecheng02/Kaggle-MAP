import os
import re
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer


def load_data(path: str) -> pd.DataFrame:
    """
    データを読み込んで返します
    Args:
        path (str): CSVファイルのパス
    Returns:
        pd.DataFrame: 読み込んだデータフレーム
    """
    df = pd.read_csv(path)
    df.Misconception = df.Misconception.fillna('NA')
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    特徴量を追加します
    """
    df['explanation_len'] = df['StudentExplanation'].fillna('').apply(len)
    df['mc_frac_count'] = df['StudentExplanation'].fillna('').apply(
        lambda x: len(re.findall(r'FRAC_\d+_\d+|\\frac', x))
    )
    df['number_count'] = df['StudentExplanation'].fillna('').apply(
        lambda x: len(re.findall(r'\b\d+\b', x))
    )
    df['operator_count'] = df['StudentExplanation'].fillna('').apply(
        lambda x: len(re.findall(r'[\+\-\*/=]', x))
    )
    df['mc_answer_len'] = df['MC_Answer'].fillna('').apply(len)
    df['question_len'] = df['QuestionText'].fillna('').apply(len)
    df['explanation_to_question_ratio'] = df['explanation_len'] / (df['question_len'] + 1)
    return df


def prepare_correct_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    正解選択肢をまとめたデータフレームを返します
    """
    idx = df.apply(lambda row: row.Category.split('_')[0]=='True', axis=1)
    correct = df.loc[idx].copy()
    correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
    correct = correct.sort_values('c', ascending=False)
    correct = correct.drop_duplicates(['QuestionId'])
    correct = correct[['QuestionId','MC_Answer']].rename(columns={'MC_Answer':'MC_Answer'})
    correct['is_correct'] = 1
    return correct

def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    カテゴリと誤解ラベルを結合して数値ラベルにエンコードします
    Args:
        df (pd.DataFrame): 元のデータフレーム
    Returns:
        tuple[pd.DataFrame, LabelEncoder]: ラベルが追加されたデータフレームと LabelEncoder
    """
    df['target'] = df['Category'] + ":" + df['Misconception']
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['target'])
    return df, le


def format_input(row: pd.Series) -> str:
    """
    モデル入力用のテキストを整形します
    """
    x = "This answer is correct." if row['is_correct'] else "This answer is incorrect."
    extra = (
        f"Additional Info: "
        f"The explanation has {row['explanation_len']} characters "
        f"and includes {row['mc_frac_count']} fraction(s)."
    )
    text = (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}\n"
        f"{extra}"
    )
    return text


def tokenize(batch: dict, tokenizer: AutoTokenizer) -> dict:
    """
    テキストをトークナイズします
    """
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=256)


def compute_map3(eval_pred) -> dict:
    """
    MAP@3を計算するカスタムメトリック
    """
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    top3 = np.argsort(-probs, axis=1)[:, :3]
    match = (top3 == labels[:, None])
    map3 = 0.0
    for i in range(len(labels)):
        if match[i,0]:
            map3 += 1.0
        elif match[i,1]:
            map3 += 1.0/2
        elif match[i,2]:
            map3 += 1.0/3
    return {"map@3": map3 / len(labels)}
