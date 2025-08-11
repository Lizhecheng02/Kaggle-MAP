# -*- coding: utf-8 -*-
"""
fork-improvements-optimized.py

Map-Charting Student Math Misunderstandings
最適化版モデル学習・予測スクリプト
"""

import numpy as np
import pandas as pd
import cudf
import cuml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics
import re
from nltk.stem import WordNetLemmatizer
import nltk
from scipy import sparse
import xgboost as xgb

# NLTK WordNetデータをダウンロード（初回実行時のみ必要）
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
import warnings
warnings.filterwarnings('ignore')

def advanced_clean(text):
    """テキストの高度なクリーニング"""
    # 数式パターンを検出
    text = re.sub(r'(\d+)\s*/\s*(\d+)', r'FRAC_\1_\2', text)
    text = re.sub(r'\\frac\{([^\}]+)\}\{([^\}]+)\}', r'FRAC_\1_\2', text)

    # 基本的なクリーニング
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s_]', '', text)

    return text.strip().lower()

def extract_math_features(text):
    """数学的特徴を抽出"""
    features = {}

    # 分数の数
    features['frac_count'] = len(re.findall(r'FRAC_\d+_\d+|\\frac', text))

    # 数値の数
    features['number_count'] = len(re.findall(r'\b\d+\b', text))

    # 演算子の数
    features['operator_count'] = len(re.findall(r'[\+\-\*\/\=]', text))

    return features

def fast_lemmatize(text):
    """単語をレンマ化"""
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def map3(target_list, pred_list):
    """MAP@3計算"""
    score = 0.
    for t, p in zip(target_list, pred_list):
        if t == p[0]:
            score += 1.
        elif len(p) > 1 and t == p[1]:
            score += 1/2
        elif len(p) > 2 and t == p[2]:
            score += 1/3
    return score / len(target_list)

def create_features(df, is_train=True):
    """特徴量作成"""
    # 基本的な長さ特徴
    df['mc_answer_len'] = df['MC_Answer'].astype(str).str.len()
    df['explanation_len'] = df['StudentExplanation'].astype(str).str.len()
    df['question_len'] = df['QuestionText'].astype(str).str.len()

    # 比率特徴
    df['explanation_to_question_ratio'] = df['explanation_len'] / (df['question_len'] + 1)

    # 数学的特徴を抽出
    for col in ['QuestionText', 'MC_Answer']:
        math_features = df[col].apply(extract_math_features).apply(pd.Series)
        prefix = 'mc_' if col == 'MC_Answer' else ''
        math_features.columns = [f'{prefix}{c}' for c in math_features.columns]
        df = pd.concat([df, math_features], axis=1)

    return df

if __name__ == '__main__':
    print('Starting optimized model training...')

    # データ読み込み
    train = pd.read_csv(
        "/kaggle/input/map-charting-student-math-misunderstandings/train.csv"
    )
    test = pd.read_csv(
        "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"
    )

    # Misconceptionの処理
    train['Misconception'] = train['Misconception'].fillna('NA').astype(str)
    train['target_cat'] = train.apply(
        lambda x: x['Category'] + ":" + x['Misconception'], axis=1
    )
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    # 特徴量作成
    train = create_features(train, is_train=True)
    test = create_features(test, is_train=False)

    # ターゲットエンコーディング
    le_target = LabelEncoder()
    train['target_encoded'] = le_target.fit_transform(train['target_cat'])
    target_classes = le_target.classes_
    n_classes = len(target_classes)
    print(f"Number of target classes: {n_classes}")

    # テキストの結合とクリーニング
    train['combined_text'] = (
        "Question: " + train['QuestionText'].astype(str) +
        " Answer: " + train['MC_Answer'].astype(str) +
        " Explanation: " + train['StudentExplanation'].astype(str)
    )
    test['combined_text'] = (
        "Question: " + test['QuestionText'].astype(str) +
        " Answer: " + test['MC_Answer'].astype(str) +
        " Explanation: " + test['StudentExplanation'].astype(str)
    )

    # テキストクリーニング
    train['cleaned_text'] = train['combined_text'].apply(advanced_clean).apply(fast_lemmatize)
    test['cleaned_text'] = test['combined_text'].apply(advanced_clean).apply(fast_lemmatize)

    # TF-IDF特徴量（scikit-learnを使用）
    print("Creating TF-IDF features...")

    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),
        max_df=0.95,
        min_df=2,
        max_features=5000  # 計算効率のため削減
    )

    # フィットと変換
    all_text = pd.concat([train['cleaned_text'], test['cleaned_text']])
    tfidf.fit(all_text)

    train_tfidf = tfidf.transform(train['cleaned_text'])
    test_tfidf = tfidf.transform(test['cleaned_text'])

    # 数値特徴量
    numeric_features = [
        'mc_answer_len', 'explanation_len', 'question_len',
        'explanation_to_question_ratio', 'frac_count', 'number_count',
        'operator_count', 'mc_frac_count', 'mc_number_count',
        'mc_operator_count'
    ]

    # 存在する特徴量のみ選択
    numeric_features = [f for f in numeric_features if f in train.columns]

    # データ準備
    X_numeric = train[numeric_features].fillna(0).values
    X_numeric_test = test[numeric_features].fillna(0).values

    # スパース行列と数値特徴を結合
    X_train_all = sparse.hstack([
        train_tfidf,
        sparse.csr_matrix(X_numeric)
    ])
    X_test_all = sparse.hstack([
        test_tfidf,
        sparse.csr_matrix(X_numeric_test)
    ])

    print(f"Final feature shape: {X_train_all.shape}")

    # XGBoostモデル
    print("Training XGBoost model...")
    nsplit=5

    skf = StratifiedKFold(n_splits=nsplit, shuffle=True, random_state=42)  # 高速化のため3分割
    oof_preds = np.zeros((len(train), n_classes))
    test_preds = np.zeros((len(test), n_classes))

    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
        'max_depth': 10,  # 8から10に増加
        'learning_rate': 0.05,  # 0.1から0.05に減少
        'subsample': 0.85,  # 0.8から0.85に増加
        'colsample_bytree': 0.85,  # 0.8から0.85に増加
        'colsample_bylevel': 0.7,  # 新規追加
        'min_child_weight': 3,  # 新規追加（過学習防止）
        'gamma': 0.1,  # 新規追加（枝刈りの閾値）
        'alpha': 0.1,  # L1正則化
        'lambda': 1.0,  # L2正則化
        'random_state': 42,
        'tree_method': 'gpu_hist',  # GPU使用
        'gpu_id': 0
    }

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train_all, train['target_encoded'])):
        print(f"Fold {fold + 1}")

        X_train_fold = X_train_all[train_idx]
        y_train_fold = train['target_encoded'].iloc[train_idx]
        X_valid_fold = X_train_all[valid_idx]
        y_valid_fold = train['target_encoded'].iloc[valid_idx]

        # DMatrix作成
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold)

        # モデル訓練
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=1000,  # 300から500に増加
            evals=[(dvalid, 'valid')],
            early_stopping_rounds=50,  # 30から50に増加
            verbose_eval=50
        )

        # 予測
        oof_preds[valid_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))
        test_preds += model.predict(xgb.DMatrix(X_test_all), iteration_range=(0, model.best_iteration)) / nsplit

    # 評価
    oof_pred_labels = np.argmax(oof_preds, axis=1)
    accuracy = np.mean(train['target_encoded'] == oof_pred_labels)
    f1 = sklearn.metrics.f1_score(train['target_encoded'], oof_pred_labels, average='weighted')

    print(f"\nValidation Accuracy: {accuracy:.4f}")
    print(f"Validation F1-score: {f1:.4f}")

    # MAP@3の計算
    top3_indices = np.argsort(-oof_preds, axis=1)[:, :3]
    predictions = []
    for indices in top3_indices:
        predictions.append([target_classes[i] for i in indices])

    map_score = map3(train['target_cat'].tolist(), predictions)
    print(f"Validation MAP@3: {map_score:.4f}")

    # テスト予測
    test_top3_indices = np.argsort(-test_preds, axis=1)[:, :3]
    test_predictions = []
    for indices in test_top3_indices:
        pred = [target_classes[i] for i in indices]
        test_predictions.append(' '.join(pred))

    # 提出ファイル作成
    submission = pd.read_csv(
        "/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv"
    )
    submission['Category:Misconception'] = test_predictions
    submission.to_csv("submission.csv", index=False)
    print("\nSubmission file created: submission_optimized.csv")
