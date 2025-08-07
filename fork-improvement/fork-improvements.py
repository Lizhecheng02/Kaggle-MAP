# -*- coding: utf-8 -*-
"""
fork-improvements.py

Map-Charting Student Math Misunderstandings
モデル学習・予測スクリプト
"""

import numpy as np
import pandas as pd
import cudf
import cuml
import sklearn
from cuml.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import sklearn.metrics
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
import re
from nltk.stem import WordNetLemmatizer
import nltk
# NLTK WordNetデータをダウンロード（初回実行時のみ必要）
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
import warnings
warnings.filterwarnings('ignore')

def fast_clean(text):
    """改行・空白・記号をクリーンし小文字化"""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip().lower()

def fast_lemmatize(text):
    """単語をステミング"""
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def map3(target_list, pred_list):
    """MAP@3計算"""
    score = 0.
    for t, p in zip(target_list, pred_list):
        if t == p[0]:
            score += 1.
        elif t == p[1]:
            score += 1/2
        elif t == p[2]:
            score += 1/3
    return score / len(target_list)

if __name__ == '__main__':
    print('RAPIDS', cuml.__version__)

    # データ読み込み
    train = pd.read_csv(
        "/kaggle/input/map-charting-student-math-misunderstandings/train.csv"
    )
    test = pd.read_csv(
        "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"
    )
    train['Misconception'] = train['Misconception'].fillna('NA').astype(str)
    train['target_cat'] = train.apply(
        lambda x: x['Category'] + ":" + x['Misconception'], axis=1
    )
    print(train.shape, test.shape)

    # ターゲットのマッピング
    map_target1 = train['Category'].value_counts().to_frame()
    map_target1['count'] = np.arange(len(map_target1))
    map_target1 = map_target1['count'].to_dict()
    map_target2 = train['Misconception'].value_counts().to_frame()
    map_target2['count'] = np.arange(len(map_target2))
    map_target2 = map_target2['count'].to_dict()

    train['target1'] = train['Category'].map(map_target1)
    train['target2'] = train['Misconception'].map(map_target2)

    # 文生成・クリーン
    train['sentence'] = (
        "Question: " + train['QuestionText'].astype(str)
        + " Answer: " + train['MC_Answer'].astype(str)
        + " Explanation: " + train['StudentExplanation'].astype(str)
    )
    test['sentence'] = (
        "Question: " + test['QuestionText'].astype(str)
        + " Answer: " + test['MC_Answer'].astype(str)
        + " Explanation: " + test['StudentExplanation'].astype(str)
    )
    train['sentence'] = train['sentence'].apply(fast_clean).apply(fast_lemmatize)
    test['sentence'] = test['sentence'].apply(fast_clean).apply(fast_lemmatize)

    # TF-IDFベクトル化 (ターゲット1用)
    tfidf1 = TfidfVectorizer(
        stop_words='english', ngram_range=(1, 4), max_df=0.95, min_df=2
    )
    tfidf1.fit(pd.concat([train['sentence'], test['sentence']]))
    train_embeddings = tfidf1.transform(train['sentence'])
    test_embeddings = tfidf1.transform(test['sentence'])
    print('Train sparse shape is', train_embeddings.shape)
    print('Test sparse shape is', test_embeddings.shape)

    # ターゲット1学習
    ytrain1 = np.zeros((len(train), len(map_target1)))
    ytest1 = np.zeros((len(test), len(map_target1)))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for i, (train_index, valid_index) in enumerate(
        skf.split(train_embeddings, train['target1'])
    ):
        print(f"Fold {i}, {len(train_index)}, {len(valid_index)}:")
        model = cuml.LogisticRegression()
        model.fit(
            train_embeddings[train_index], train['target1'].iloc[train_index]
        )
        ytrain1[valid_index] = model.predict_proba(
            train_embeddings[valid_index]
        ).get()
        ytest1 += model.predict_proba(test_embeddings).get() / 10.
    print(
        "ACC:", np.mean(train['target1'] == np.argmax(ytrain1, 1))
    )
    print(
        "F1:",
        sklearn.metrics.f1_score(
            train['target1'], np.argmax(ytrain1, 1), average='weighted'
        )
    )

    # TF-IDFベクトル化 (ターゲット2用)
    tfidf2 = TfidfVectorizer(
        stop_words='english', ngram_range=(1, 3), max_df=0.95, min_df=2
    )
    tfidf2.fit(pd.concat([train['sentence'], test['sentence']]))
    train_embeddings2 = tfidf2.transform(train['sentence'])
    test_embeddings2 = tfidf2.transform(test['sentence'])
    print('Train sparse shape is', train_embeddings2.shape)
    print('Test sparse shape is', test_embeddings2.shape)

    # ターゲット2学習
    ytrain2 = np.zeros((len(train), len(map_target2)))
    ytest2 = np.zeros((len(test), len(map_target2)))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for i, (train_index, valid_index) in enumerate(
        skf.split(train_embeddings2, train['target2'])
    ):
        print(f"Fold {i}, {len(train_index)}, {len(valid_index)}:")
        model = cuml.LogisticRegression(class_weight='balanced')
        model.fit(
            train_embeddings2[train_index], train['target2'].iloc[train_index]
        )
        ytrain2[valid_index] = model.predict_proba(
            train_embeddings2[valid_index]
        ).get()
        ytest2 += model.predict_proba(test_embeddings2).get() / 10.
    print(
        "ACC:", np.mean(train['target2'] == np.argmax(ytrain2, 1))
    )
    print(
        "F1:",
        sklearn.metrics.f1_score(
            train['target2'], np.argmax(ytrain2, 1), average='weighted'
        )
    )

    # 逆マッピング
    map_inverse1 = {v: k for k, v in map_target1.items()}
    map_inverse2 = {v: k for k, v in map_target2.items()}

    # 上位3予測取得
    ytrain2[:, 0] = 0
    predicted1 = np.argsort(-ytrain1, 1)[:, :3]
    predicted2 = np.argsort(-ytrain2, 1)[:, :3]
    predict = []
    for i in range(len(predicted1)):
        preds = []
        for j in range(3):
            p1 = map_inverse1[predicted1[i, j]]
            p2 = map_inverse2[predicted2[i, j]]
            if 'Misconception' in p1:
                preds.append(f"{p1}:{p2}")
            else:
                preds.append(f"{p1}:NA")
        predict.append(preds)
    print(f"MAP@3: {map3(train['target_cat'].tolist(), predict)}")

    # テストデータ予測 & 提出
    ytest2[:, 0] = 0
    predicted1 = np.argsort(-ytest1, 1)[:, :3]
    predicted2 = np.argsort(-ytest2, 1)[:, :3]
    test_preds = []
    for i in range(len(predicted1)):
        preds = []
        for j in range(3):
            p1 = map_inverse1[predicted1[i, j]]
            p2 = map_inverse2[predicted2[i, j]]
            if 'Misconception' in p1:
                preds.append(f"{p1}:{p2}")
            else:
                preds.append(f"{p1}:NA")
        test_preds.append(' '.join(preds))
    sub = pd.read_csv(
        "/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv"
    )
    sub['Category:Misconception'] = test_preds
    sub.to_csv("submission.csv", index=False)
