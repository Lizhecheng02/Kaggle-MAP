# -*- coding: utf-8 -*-
"""
fork-improvements-enhanced.py

Map-Charting Student Math Misunderstandings
改善版モデル学習・予測スクリプト
"""

import numpy as np
import pandas as pd
import cudf
import cuml
from cuml.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics
import re
from nltk.stem import WordNetLemmatizer
import nltk
from scipy import sparse
import xgboost as xgb
import lightgbm as lgb
import cupy as cp
import torch

# NLTK WordNetデータをダウンロード（初回実行時のみ必要）
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
import warnings
warnings.filterwarnings('ignore')

# GPU利用可能性のチェック
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"CuPy version: {cp.__version__}")

def advanced_clean(text):
    """テキストの高度なクリーニング"""
    # LaTeX数式を保護
    math_patterns = re.findall(r'\$[^\$]+\$|\\\([^\)]+\\\)|\\\[[^\]]+\\\]', text)
    for i, pattern in enumerate(math_patterns):
        text = text.replace(pattern, f' MATH{i} ')
    
    # 分数のパターンを検出
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
    
    # パーセント記号
    features['percent_count'] = text.count('%')
    
    # 括弧の数
    features['parenthesis_count'] = text.count('(') + text.count(')')
    
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
    # cuDFかpandasかを判定
    is_cudf = hasattr(df, 'to_pandas')
    
    if is_cudf:
        # cuDFでの処理
        df['mc_answer_len'] = df['MC_Answer'].str.len()
        df['explanation_len'] = df['StudentExplanation'].str.len()
        df['question_len'] = df['QuestionText'].str.len()
    else:
        # pandasでの処理
        df['mc_answer_len'] = df['MC_Answer'].astype(str).str.len()
        df['explanation_len'] = df['StudentExplanation'].astype(str).str.len()
        df['question_len'] = df['QuestionText'].astype(str).str.len()
    
    # 比率特徴
    df['explanation_to_question_ratio'] = df['explanation_len'] / (df['question_len'] + 1)
    
    # 数学的特徴を抽出（pandas DataFrameで処理）
    if is_cudf:
        df_pd = df.to_pandas()
        math_features = df_pd['QuestionText'].apply(extract_math_features).apply(pd.Series)
        mc_math_features = df_pd['MC_Answer'].apply(extract_math_features).apply(pd.Series)
        mc_math_features.columns = [f'mc_{col}' for col in mc_math_features.columns]
        
        # 結果をcuDFに変換して結合
        df = cudf.concat([df, cudf.from_pandas(math_features)], axis=1)
        df = cudf.concat([df, cudf.from_pandas(mc_math_features)], axis=1)
    else:
        math_features = df['QuestionText'].apply(extract_math_features).apply(pd.Series)
        df = pd.concat([df, math_features], axis=1)
        
        mc_math_features = df['MC_Answer'].apply(extract_math_features).apply(pd.Series)
        mc_math_features.columns = [f'mc_{col}' for col in mc_math_features.columns]
        df = pd.concat([df, mc_math_features], axis=1)
    
    return df

if __name__ == '__main__':
    print('Starting enhanced model training...')
    
    # データ読み込み（cudfを使用してGPU上で読み込み）
    train = cudf.read_csv(
        "/kaggle/input/map-charting-student-math-misunderstandings/train.csv"
    )
    test = cudf.read_csv(
        "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"
    )
    
    # 一部の処理でpandasが必要な場合のため、バックアップを作成
    train_pd = train.to_pandas()
    test_pd = test.to_pandas()
    
    # Misconceptionの処理（pandas DataFrameで処理）
    train_pd['Misconception'] = train_pd['Misconception'].fillna('NA').astype(str)
    train_pd['target_cat'] = train_pd.apply(
        lambda x: x['Category'] + ":" + x['Misconception'], axis=1
    )
    # cuDFに反映
    train['Misconception'] = cudf.from_pandas(train_pd['Misconception'])
    train['target_cat'] = cudf.from_pandas(train_pd['target_cat'])
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    
    # 特徴量作成（cuDFのまま処理）
    train = create_features(train, is_train=True)
    test = create_features(test, is_train=False)
    
    # 一部の処理でpandasが必要なため、更新
    train_pd = train.to_pandas()
    test_pd = test.to_pandas()
    
    # ターゲットエンコーディング（pandas DataFrameで処理）
    le_target = LabelEncoder()
    train_pd['target_encoded'] = le_target.fit_transform(train_pd['target_cat'])
    target_classes = le_target.classes_
    n_classes = len(target_classes)
    print(f"Number of target classes: {n_classes}")
    
    # cuDFに反映
    train['target_encoded'] = cudf.from_pandas(train_pd['target_encoded'])
    
    # テキストの結合とクリーニング（pandas DataFrameで処理）
    train_pd['combined_text'] = (
        "Question: " + train_pd['QuestionText'].astype(str) + 
        " Answer: " + train_pd['MC_Answer'].astype(str) + 
        " Explanation: " + train_pd['StudentExplanation'].astype(str)
    )
    test_pd['combined_text'] = (
        "Question: " + test_pd['QuestionText'].astype(str) + 
        " Answer: " + test_pd['MC_Answer'].astype(str) + 
        " Explanation: " + test_pd['StudentExplanation'].astype(str)
    )
    
    # テキストクリーニング（pandas DataFrameで処理）
    train_pd['cleaned_text'] = train_pd['combined_text'].apply(advanced_clean).apply(fast_lemmatize)
    test_pd['cleaned_text'] = test_pd['combined_text'].apply(advanced_clean).apply(fast_lemmatize)
    
    # QuestionIdによるグループ特徴（pandas DataFrameで処理）
    question_stats = train_pd.groupby('QuestionId').agg({
        'target_cat': ['count', 'nunique'],
        'Category': lambda x: x.value_counts().index[0]  # 最頻値
    }).reset_index()
    question_stats.columns = ['QuestionId', 'question_count', 'question_unique_targets', 'question_mode_category']
    
    train_pd = train_pd.merge(question_stats, on='QuestionId', how='left')
    test_pd = test_pd.merge(question_stats, on='QuestionId', how='left')
    
    # TF-IDF特徴量（複数のn-gramレンジ）
    print("Creating TF-IDF features...")
    
    # 単語レベル
    tfidf_word = TfidfVectorizer(
        stop_words='english', 
        ngram_range=(1, 3), 
        max_df=0.95, 
        min_df=2,
        max_features=10000
    )
    
    # 文字レベル
    tfidf_char = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 5),
        max_df=0.95,
        min_df=2,
        max_features=5000
    )
    
    # フィット
    all_text = pd.concat([train_pd['cleaned_text'], test_pd['cleaned_text']])
    tfidf_word.fit(all_text)
    tfidf_char.fit(all_text)
    
    # 変換（GPU上で保持）
    train_tfidf_word = tfidf_word.transform(train_pd['cleaned_text'])
    test_tfidf_word = tfidf_word.transform(test_pd['cleaned_text'])
    
    train_tfidf_char = tfidf_char.transform(train_pd['cleaned_text'])
    test_tfidf_char = tfidf_char.transform(test_pd['cleaned_text'])
    
    # 数値特徴量
    numeric_features = [
        'mc_answer_len', 'explanation_len', 'question_len',
        'explanation_to_question_ratio', 'frac_count', 'number_count',
        'operator_count', 'percent_count', 'parenthesis_count',
        'mc_frac_count', 'mc_number_count', 'mc_operator_count',
        'mc_percent_count', 'mc_parenthesis_count'
    ]
    
    # 存在する特徴量のみ選択
    numeric_features = [f for f in numeric_features if f in train_pd.columns]
    
    # LightGBMモデル
    print("Training LightGBM model...")
    
    # データ準備（GPU上で処理）
    X_numeric = cp.asarray(train_pd[numeric_features].fillna(0).values)
    X_numeric_test = cp.asarray(test_pd[numeric_features].fillna(0).values)
    
    # GPU上でスパース行列と数値特徴を結合
    from cupyx.scipy.sparse import hstack as gpu_hstack
    from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix
    
    X_train_all = gpu_hstack([
        train_tfidf_word, 
        train_tfidf_char,
        gpu_csr_matrix(X_numeric)
    ]).tocsr()  # CSR形式に変換
    X_test_all = gpu_hstack([
        test_tfidf_word,
        test_tfidf_char,
        gpu_csr_matrix(X_numeric_test)
    ]).tocsr()  # CSR形式に変換
    
    print(f"Final feature shape: {X_train_all.shape}")
    
    # Cross-validation with LightGBM
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(train), n_classes))
    test_preds = np.zeros((len(test), n_classes))
    
    lgb_params = {
        'objective': 'multiclass',
        'num_class': n_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 64,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }
    
    # StratifiedKFoldのために一時的にCPUに転送（形状情報のみ必要）
    dummy_X = np.zeros((X_train_all.shape[0], 1))
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(dummy_X, train_pd['target_encoded'])):
        print(f"Fold {fold + 1}")
        
        # GPU上のデータから必要な部分を取得してCPUに転送
        X_train_fold = X_train_all[train_idx].get()
        y_train_fold = train_pd['target_encoded'].iloc[train_idx]
        X_valid_fold = X_train_all[valid_idx].get()
        y_valid_fold = train_pd['target_encoded'].iloc[valid_idx]
        
        # LightGBMデータセット作成
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        valid_data = lgb.Dataset(X_valid_fold, label=y_valid_fold, reference=train_data)
        
        # モデル訓練
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        # 予測
        oof_preds[valid_idx] = model.predict(X_valid_fold, num_iteration=model.best_iteration)
        test_preds += model.predict(X_test_all.get(), num_iteration=model.best_iteration) / 5
    
    # 評価
    oof_pred_labels = np.argmax(oof_preds, axis=1)
    accuracy = np.mean(train_pd['target_encoded'] == oof_pred_labels)
    f1 = sklearn.metrics.f1_score(train_pd['target_encoded'], oof_pred_labels, average='weighted')
    
    print(f"\nValidation Accuracy: {accuracy:.4f}")
    print(f"Validation F1-score: {f1:.4f}")
    
    # MAP@3の計算
    top3_indices = np.argsort(-oof_preds, axis=1)[:, :3]
    predictions = []
    for indices in top3_indices:
        predictions.append([target_classes[i] for i in indices])
    
    map_score = map3(train_pd['target_cat'].tolist(), predictions)
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
    submission.to_csv("submission_enhanced.csv", index=False)
    print("\nSubmission file created: submission_enhanced.csv")