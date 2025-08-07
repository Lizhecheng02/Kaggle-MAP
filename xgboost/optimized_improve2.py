# -*- coding: utf-8 -*-
"""
optimized_improve2.py

Map-Charting Student Math Misunderstandings
改良版モデル学習・予測スクリプト（強化版）
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
from collections import Counter
import string

# NLTK WordNetデータをダウンロード（初回実行時のみ必要）
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

def advanced_clean(text):
    """テキストの高度なクリーニング"""
    # 数式パターンを検出
    text = re.sub(r'(\d+)\s*/\s*(\d+)', r'FRAC_\1_\2', text)
    text = re.sub(r'\\frac\{([^\}]+)\}\{([^\}]+)\}', r'FRAC_\1_\2', text)

    # 数学記号の検出
    text = re.sub(r'(\d+)\s*\+\s*(\d+)', r'\1 PLUS \2', text)
    text = re.sub(r'(\d+)\s*\-\s*(\d+)', r'\1 MINUS \2', text)
    text = re.sub(r'(\d+)\s*\*\s*(\d+)', r'\1 TIMES \2', text)
    text = re.sub(r'(\d+)\s*÷\s*(\d+)', r'\1 DIVIDE \2', text)
    text = re.sub(r'=', ' EQUALS ', text)

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
    features['operator_count'] = len(re.findall(r'[\+\-\*\/\=]|PLUS|MINUS|TIMES|DIVIDE|EQUALS', text))

    # 括弧の数
    features['parenthesis_count'] = len(re.findall(r'[\(\)]', text))

    # 小数点の数
    features['decimal_count'] = len(re.findall(r'\d+\.\d+', text))

    # 変数の数（単一文字）
    features['variable_count'] = len(re.findall(r'\b[a-zA-Z]\b', text))

    # 数学用語の数
    math_terms = ['equation', 'solve', 'calculate', 'formula', 'variable', 'fraction',
                  'decimal', 'percent', 'ratio', 'proportion', 'sum', 'difference',
                  'product', 'quotient', 'remainder', 'factor', 'multiple']
    features['math_term_count'] = sum(1 for term in math_terms if term in text.lower())

    return features

def extract_statistical_features(text):
    """統計的特徴を抽出"""
    features = {}

    # 単語数
    words = text.split()
    features['word_count'] = len(words)

    # 平均単語長
    features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0

    # ユニークな単語の割合
    features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0

    # 文の数（簡易的）
    features['sentence_count'] = len(re.split(r'[.!?]+', text))

    # 大文字の割合（元のテキストに対して）
    original_text = text
    features['uppercase_ratio'] = sum(1 for c in original_text if c.isupper()) / len(original_text) if original_text else 0

    return features

def fast_lemmatize(text):
    """単語をレンマ化"""
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def extract_error_patterns(text):
    """エラーパターンを抽出"""
    patterns = {}

    # 一般的な数学的誤解パターン
    patterns['addition_subtraction_confusion'] = int('add' in text and 'subtract' in text)
    patterns['multiplication_division_confusion'] = int('multiply' in text and 'divide' in text)
    patterns['order_of_operations'] = int('order' in text or 'first' in text or 'then' in text)
    patterns['fraction_decimal_confusion'] = int('fraction' in text and 'decimal' in text)
    patterns['negative_number_issue'] = int('negative' in text or 'minus' in text)

    return patterns

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
    df['answer_to_question_ratio'] = df['mc_answer_len'] / (df['question_len'] + 1)
    df['explanation_to_answer_ratio'] = df['explanation_len'] / (df['mc_answer_len'] + 1)

    # 数学的特徴を抽出
    for col in ['QuestionText', 'MC_Answer', 'StudentExplanation']:
        math_features = df[col].apply(extract_math_features).apply(pd.Series)
        prefix = 'q_' if col == 'QuestionText' else ('mc_' if col == 'MC_Answer' else 'exp_')
        math_features.columns = [f'{prefix}{c}' for c in math_features.columns]
        df = pd.concat([df, math_features], axis=1)

    # 統計的特徴を抽出
    for col in ['StudentExplanation']:
        stat_features = df[col].apply(extract_statistical_features).apply(pd.Series)
        stat_features.columns = [f'exp_{c}' for c in stat_features.columns]
        df = pd.concat([df, stat_features], axis=1)

    # エラーパターン特徴
    error_features = df['StudentExplanation'].apply(extract_error_patterns).apply(pd.Series)
    error_features.columns = [f'error_{c}' for c in error_features.columns]
    df = pd.concat([df, error_features], axis=1)

    # 相互作用特徴
    df['total_numbers'] = df['q_number_count'] + df['mc_number_count'] + df['exp_number_count']
    df['total_operators'] = df['q_operator_count'] + df['mc_operator_count'] + df['exp_operator_count']
    df['math_complexity'] = df['total_numbers'] * df['total_operators']

    return df

def post_process_predictions(predictions, test_df):
    """予測の後処理"""
    # 簡単な問題（数値が少ない）の場合、信頼度を調整
    simple_mask = test_df['q_number_count'] <= 2

    # カテゴリごとの頻度に基づく調整も可能
    # ここでは実装をシンプルに保つ

    return predictions

if __name__ == '__main__':
    print('Starting improved model training (v2)...')

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

    # TF-IDF特徴量（パラメータ調整）
    print("Creating TF-IDF features...")

    # カスタムストップワード
    stop_words = list(stopwords.words('english'))
    stop_words.extend(['question', 'answer', 'explanation', 'student'])

    tfidf = TfidfVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 4),  # 4-gramまで拡張
        max_df=0.9,
        min_df=2,
        max_features=10000,  # 8000→10000に増やす
        token_pattern=r'\b\w+\b',
        sublinear_tf=True  # TF-IDFのスケーリング
    )

    # フィットと変換
    all_text = pd.concat([train['cleaned_text'], test['cleaned_text']])
    tfidf.fit(all_text)

    train_tfidf = tfidf.transform(train['cleaned_text'])
    test_tfidf = tfidf.transform(test['cleaned_text'])

    # 数値特徴量（拡張版）
    numeric_features = [
        'mc_answer_len', 'explanation_len', 'question_len',
        'explanation_to_question_ratio', 'answer_to_question_ratio',
        'explanation_to_answer_ratio',
        'q_frac_count', 'q_number_count', 'q_operator_count',
        'q_parenthesis_count', 'q_decimal_count', 'q_variable_count', 'q_math_term_count',
        'mc_frac_count', 'mc_number_count', 'mc_operator_count',
        'mc_parenthesis_count', 'mc_decimal_count', 'mc_variable_count', 'mc_math_term_count',
        'exp_frac_count', 'exp_number_count', 'exp_operator_count',
        'exp_parenthesis_count', 'exp_decimal_count', 'exp_variable_count', 'exp_math_term_count',
        'exp_word_count', 'exp_avg_word_length', 'exp_unique_word_ratio',
        'exp_sentence_count', 'exp_uppercase_ratio',
        'error_addition_subtraction_confusion', 'error_multiplication_division_confusion',
        'error_order_of_operations', 'error_fraction_decimal_confusion',
        'error_negative_number_issue',
        'total_numbers', 'total_operators', 'math_complexity'
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

    # XGBoostモデル（複数の設定でアンサンブル）
    print("Training XGBoost ensemble...")
    nsplit = 5

    skf = StratifiedKFold(n_splits=nsplit, shuffle=True, random_state=42)

    # 複数モデルの予測を保存
    model_configs = [
        # モデル1: より深く、より強い正則化
        {
            'objective': 'multi:softprob',
            'num_class': n_classes,
            'eval_metric': 'mlogloss',
            'max_depth': 16,  # 14→16に深くする
            'learning_rate': 0.025,  # 0.03→0.025に下げて過学習を防ぐ
            'subsample': 0.85,  # 0.9→0.85
            'colsample_bytree': 0.85,  # 0.9→0.85
            'colsample_bylevel': 0.5,  # 0.6→0.5
            'min_child_weight': 3,  # 2→3
            'gamma': 0.2,  # 0.15→0.2
            'alpha': 0.2,  # 0.1→0.2
            'lambda': 2.0,  # 1.5→2.0
            'max_delta_step': 1,  # 新規追加：勾配のステップサイズ制限
            'scale_pos_weight': 1.2,  # 新規追加：不均衡データ対策
            'random_state': 42,
            'tree_method': 'gpu_hist',
            'gpu_id': 0
        },
        # モデル2: 浅めで高速学習
        {
            'objective': 'multi:softprob',
            'num_class': n_classes,
            'eval_metric': 'mlogloss',
            'max_depth': 10,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'min_child_weight': 1,
            'gamma': 0.05,
            'alpha': 0.05,
            'lambda': 1.0,
            'random_state': 123,
            'tree_method': 'gpu_hist',
            'gpu_id': 0
        },
        # モデル3: ダートブースター（異なるアルゴリズム）
        {
            'booster': 'dart',  # DARTブースター
            'objective': 'multi:softprob',
            'num_class': n_classes,
            'eval_metric': 'mlogloss',
            'max_depth': 12,
            'learning_rate': 0.04,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'rate_drop': 0.1,  # DART専用パラメータ
            'skip_drop': 0.5,  # DART専用パラメータ
            'min_child_weight': 2,
            'gamma': 0.1,
            'alpha': 0.15,
            'lambda': 1.5,
            'random_state': 456,
            'tree_method': 'gpu_hist',
            'gpu_id': 0
        }
    ]

    all_oof_preds = []
    all_test_preds = []

    for config_idx, xgb_params in enumerate(model_configs):
        print(f"\n{'='*50}")
        print(f"Training model configuration {config_idx + 1}/{len(model_configs)}")
        print(f"Model type: {xgb_params.get('booster', 'gbtree')}")
        print(f"Max depth: {xgb_params.get('max_depth', 'N/A')}")
        print(f"Learning rate: {xgb_params.get('learning_rate', 'N/A')}")
        print(f"{'='*50}")

        oof_preds = np.zeros((len(train), n_classes))
        test_preds = np.zeros((len(test), n_classes))

        for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train_all, train['target_encoded'])):
            print(f"\nFold {fold + 1}/{nsplit}")

            X_train_fold = X_train_all[train_idx]
            y_train_fold = train['target_encoded'].iloc[train_idx]
            X_valid_fold = X_train_all[valid_idx]
            y_valid_fold = train['target_encoded'].iloc[valid_idx]

            # DMatrix作成
            dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
            dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold)

            # 学習率スケジューリング用のコールバック関数
            def learning_rate_schedule(boosting_round):
                """学習率を徐々に下げる"""
                base_lr = xgb_params.get('learning_rate', 0.03)
                # 1000ラウンドまでは元の学習率、その後徐々に減衰
                if boosting_round < 1000:
                    return base_lr
                else:
                    # 0.99の累乗で減衰、最小値0.01
                    return max(0.01, base_lr * (0.995 ** (boosting_round - 1000)))
            
            # コールバック設定
            callbacks = [
                xgb.callback.LearningRateScheduler(learning_rate_schedule)
            ]

            # モデル訓練（パラメータ調整）
            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=2000,  # 1500→2000
                evals=[(dvalid, 'valid')],
                early_stopping_rounds=150,  # 100→150
                verbose_eval=100,
                callbacks=callbacks
            )

            print(f"Best iteration: {model.best_iteration}")

            # 予測
            oof_preds[valid_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))
            test_preds += model.predict(xgb.DMatrix(X_test_all), iteration_range=(0, model.best_iteration)) / nsplit

        # モデルごとのスコア計算
        model_oof_pred_labels = np.argmax(oof_preds, axis=1)
        model_accuracy = np.mean(train['target_encoded'] == model_oof_pred_labels)
        model_f1 = sklearn.metrics.f1_score(train['target_encoded'], model_oof_pred_labels, average='weighted')

        # MAP@3の計算
        model_top3_indices = np.argsort(-oof_preds, axis=1)[:, :3]
        model_predictions = []
        for indices in model_top3_indices:
            model_predictions.append([target_classes[i] for i in indices])

        model_map_score = map3(train['target_cat'].tolist(), model_predictions)

        print(f"\nModel {config_idx + 1} Validation Metrics:")
        print(f"  Accuracy: {model_accuracy:.4f}")
        print(f"  F1-score: {model_f1:.4f}")
        print(f"  MAP@3: {model_map_score:.4f}")

        all_oof_preds.append(oof_preds)
        all_test_preds.append(test_preds)

    # アンサンブル（平均）
    print(f"\n{'='*50}")
    print("Ensemble Results")
    print(f"{'='*50}")

    final_oof_preds = np.mean(all_oof_preds, axis=0)
    final_test_preds = np.mean(all_test_preds, axis=0)

    # 最終評価
    oof_pred_labels = np.argmax(final_oof_preds, axis=1)
    accuracy = np.mean(train['target_encoded'] == oof_pred_labels)
    f1 = sklearn.metrics.f1_score(train['target_encoded'], oof_pred_labels, average='weighted')

    print(f"\nFinal Ensemble Validation Accuracy: {accuracy:.4f}")
    print(f"Final Ensemble Validation F1-score: {f1:.4f}")

    # MAP@3の計算
    top3_indices = np.argsort(-final_oof_preds, axis=1)[:, :3]
    predictions = []
    for indices in top3_indices:
        predictions.append([target_classes[i] for i in indices])

    map_score = map3(train['target_cat'].tolist(), predictions)
    print(f"Final Ensemble Validation MAP@3: {map_score:.4f}")

    # テスト予測（後処理付き）
    final_test_preds = post_process_predictions(final_test_preds, test)

    test_top3_indices = np.argsort(-final_test_preds, axis=1)[:, :3]
    test_predictions = []
    for indices in test_top3_indices:
        pred = [target_classes[i] for i in indices]
        test_predictions.append(' '.join(pred))

    # 提出ファイル作成
    submission = pd.read_csv(
        "/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv"
    )
    submission['Category:Misconception'] = test_predictions
    submission.to_csv("submission_improved_v2.csv", index=False)
    print("\nSubmission file created: submission_improved_v2.csv")