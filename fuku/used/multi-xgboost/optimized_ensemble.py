# -*- coding: utf-8 -*-
"""
optimized_ensemble.py

Map-Charting Student Math Misunderstandings
複数のXGBoostモデルをアンサンブルする版
"""

import numpy as np
import pandas as pd
import cudf
import cuml
from cuml.feature_extraction.text import TfidfVectorizer as cuTfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics
import re
from multiprocessing import Pool, cpu_count
import xgboost as xgb
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

# GPU設定
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def clean_batch(batch):
    """バッチ単位でテキストクリーニング"""
    cleaned = []
    for text in batch:
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

        cleaned.append(text.strip().lower())
    return cleaned

def parallel_clean(texts, n_jobs=None):
    """並列でテキストクリーニング"""
    if n_jobs is None:
        n_jobs = cpu_count()

    # バッチに分割
    batch_size = len(texts) // n_jobs + 1
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

    # 並列処理
    with Pool(n_jobs) as pool:
        results = pool.map(clean_batch, batches)

    # 結果を結合
    cleaned_texts = []
    for batch_result in results:
        cleaned_texts.extend(batch_result)

    return cleaned_texts

def extract_features_batch(df, is_train=True):
    """バッチで特徴量抽出（GPU活用）"""
    # cuDFに変換してGPU上で処理
    gdf = cudf.from_pandas(df)

    # 基本的な長さ特徴（GPU上で計算）
    gdf['mc_answer_len'] = gdf['MC_Answer'].astype(str).str.len()
    gdf['explanation_len'] = gdf['StudentExplanation'].astype(str).str.len()
    gdf['question_len'] = gdf['QuestionText'].astype(str).str.len()

    # 比率特徴（GPU上で計算）
    gdf['explanation_to_question_ratio'] = gdf['explanation_len'] / (gdf['question_len'] + 1)
    gdf['answer_to_question_ratio'] = gdf['mc_answer_len'] / (gdf['question_len'] + 1)
    gdf['explanation_to_answer_ratio'] = gdf['explanation_len'] / (gdf['mc_answer_len'] + 1)

    # pandasに戻す（必要な部分のみ）
    df_features = gdf.to_pandas()

    # 数学的特徴を並列抽出
    for col in ['QuestionText', 'MC_Answer', 'StudentExplanation']:
        features = extract_math_features_parallel(df[col].values)
        prefix = 'q_' if col == 'QuestionText' else ('mc_' if col == 'MC_Answer' else 'exp_')
        for feat_name, feat_values in features.items():
            df_features[f'{prefix}{feat_name}'] = feat_values

    return df_features

def extract_math_features_parallel(texts):
    """並列で数学的特徴を抽出"""
    def extract_single(text):
        text = str(text)
        return {
            'frac_count': len(re.findall(r'FRAC_\d+_\d+|\\frac', text)),
            'number_count': len(re.findall(r'\b\d+\b', text)),
            'operator_count': len(re.findall(r'[\+\-\*\/\=]|PLUS|MINUS|TIMES|DIVIDE|EQUALS', text)),
            'parenthesis_count': len(re.findall(r'[\(\)]', text)),
            'decimal_count': len(re.findall(r'\d+\.\d+', text)),
            'variable_count': len(re.findall(r'\b[a-zA-Z]\b', text)),
        }

    # バッチ処理で高速化
    results = [extract_single(text) for text in texts]

    # 結果を辞書形式に変換
    features = {}
    for key in results[0].keys():
        features[key] = [r[key] for r in results]

    return features

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

# 複数モデルの設定
def get_model_configs(n_classes):
    """複数のXGBoostモデル設定を返す"""
    model_configs = [
        # モデル1: より深く、より強い正則化
        {
            'name': 'Deep_Regularized',
            'params': {
                'objective': 'multi:softprob',
                'num_class': n_classes,
                'eval_metric': 'mlogloss',
                'max_depth': 24,
                'learning_rate': 0.025,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'colsample_bylevel': 0.5,
                'min_child_weight': 3,
                'gamma': 0.2,
                'alpha': 0.2,
                'lambda': 2.0,
                'max_delta_step': 1,
                'scale_pos_weight': 1.2,
                'random_state': 42,
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor',
                'gpu_id': 0,
                'max_bin': 2048,
                'grow_policy': 'depthwise',
                'single_precision_histogram': True,
                'updater': 'grow_gpu_hist',
                'sampling_method': 'gradient_based',
            },
            'num_boost_round': 1500,
            'early_stopping_rounds': 100
        },
        # モデル2: 浅めで高速学習
        # {
        #     'name': 'Fast_Shallow',
        #     'params': {
        #         'objective': 'multi:softprob',
        #         'num_class': n_classes,
        #         'eval_metric': 'mlogloss',
        #         'max_depth': 10,
        #         'learning_rate': 0.05,
        #         'subsample': 0.9,
        #         'colsample_bytree': 0.8,
        #         'colsample_bylevel': 0.8,
        #         'min_child_weight': 1,
        #         'gamma': 0.05,
        #         'alpha': 0.05,
        #         'lambda': 1.0,
        #         'random_state': 123,
        #         'tree_method': 'gpu_hist',
        #         'predictor': 'gpu_predictor',
        #         'gpu_id': 0,
        #         'max_bin': 512,
        #         'grow_policy': 'depthwise',
        #         'single_precision_histogram': True,
        #         'updater': 'grow_gpu_hist',
        #     },
        #     'num_boost_round': 1000,
        #     'early_stopping_rounds': 80
        # },
        # # モデル3: 高速で効率的なモデル
        # {
        #     'name': 'Efficient_Model',
        #     'params': {
        #         'objective': 'multi:softprob',
        #         'num_class': n_classes,
        #         'eval_metric': 'mlogloss',
        #         'max_depth': 12,
        #         'learning_rate': 0.04,
        #         'subsample': 0.8,
        #         'colsample_bytree': 0.7,
        #         'colsample_bylevel': 0.7,
        #         'min_child_weight': 2,
        #         'gamma': 0.1,
        #         'alpha': 0.15,
        #         'lambda': 1.5,
        #         'random_state': 456,
        #         'tree_method': 'gpu_hist',
        #         'predictor': 'gpu_predictor',
        #         'gpu_id': 0,
        #         'max_bin': 256,
        #         'grow_policy': 'lossguide',
        #         'max_leaves': 63,
        #         'single_precision_histogram': True,
        #         'updater': 'grow_gpu_hist',
        #         'sampling_method': 'gradient_based',
        #     },
        #     'num_boost_round': 1100,
        #     'early_stopping_rounds': 80
        # },
        # # モデル4: 中程度の深さで異なる正則化
        # {
        #     'name': 'Medium_Balanced',
        #     'params': {
        #         'objective': 'multi:softprob',
        #         'num_class': n_classes,
        #         'eval_metric': 'mlogloss',
        #         'max_depth': 14,
        #         'learning_rate': 0.035,
        #         'subsample': 0.75,
        #         'colsample_bytree': 0.75,
        #         'colsample_bylevel': 0.6,
        #         'min_child_weight': 2,
        #         'gamma': 0.15,
        #         'alpha': 0.1,
        #         'lambda': 1.8,
        #         'random_state': 789,
        #         'tree_method': 'gpu_hist',
        #         'predictor': 'gpu_predictor',
        #         'gpu_id': 0,
        #         'max_bin': 896,
        #         'grow_policy': 'depthwise',
        #         'single_precision_histogram': True,
        #         'updater': 'grow_gpu_hist',
        #     },
        #     'num_boost_round': 1200,
        #     'early_stopping_rounds': 90
        # }
    ]
    return model_configs

if __name__ == '__main__':
    print('Starting Multi-XGBoost Ensemble model training...')

    # GPU使用状況の確認
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"GPU devices available: {device_count}")
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        print(f"GPU {i}: {name}")
        print(f"  Memory: {info.used / 1024**3:.1f}GB / {info.total / 1024**3:.1f}GB")

    # データ読み込み（GPU高速化）
    print("Loading data with cuDF...")
    train = cudf.read_csv(
        "/kaggle/input/map-charting-student-math-misunderstandings/train.csv"
    ).to_pandas()

    test = cudf.read_csv(
        "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"
    ).to_pandas()

    # Misconceptionの処理
    train['Misconception'] = train['Misconception'].fillna('NA').astype(str)
    train['target_cat'] = train.apply(
        lambda x: x['Category'] + ":" + x['Misconception'], axis=1
    )
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    # テキストの結合
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

    # 並列テキストクリーニング
    print("Parallel text cleaning...")
    train_texts = train['combined_text'].tolist()
    test_texts = test['combined_text'].tolist()

    train['cleaned_text'] = parallel_clean(train_texts, n_jobs=8)
    test['cleaned_text'] = parallel_clean(test_texts, n_jobs=8)

    # 特徴量作成（GPU活用）
    print("Extracting features with GPU acceleration...")
    train_features = extract_features_batch(train, is_train=True)
    test_features = extract_features_batch(test, is_train=False)

    # 元のデータフレームに特徴量を追加
    for col in train_features.columns:
        if col not in ['QuestionText', 'MC_Answer', 'StudentExplanation', 'Category', 'Misconception', 'target_cat', 'combined_text', 'cleaned_text']:
            train[col] = train_features[col]

    for col in test_features.columns:
        if col not in ['QuestionText', 'MC_Answer', 'StudentExplanation', 'combined_text', 'cleaned_text']:
            test[col] = test_features[col]

    # ターゲットエンコーディング
    le_target = LabelEncoder()
    train['target_encoded'] = le_target.fit_transform(train['target_cat'])
    target_classes = le_target.classes_
    n_classes = len(target_classes)
    print(f"Number of target classes: {n_classes}")

    # TF-IDF特徴量
    print("Creating TF-IDF features...")
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_df=0.9,
        min_df=2,
        max_features=5000,
        token_pattern=r'\b\w+\b',
        sublinear_tf=True
    )

    # フィットと変換
    all_text = pd.concat([train['cleaned_text'], test['cleaned_text']])
    tfidf.fit(all_text)

    train_tfidf = tfidf.transform(train['cleaned_text'])
    test_tfidf = tfidf.transform(test['cleaned_text'])

    # 数値特徴量
    numeric_features = [
        'mc_answer_len', 'explanation_len', 'question_len',
        'explanation_to_question_ratio', 'answer_to_question_ratio',
        'explanation_to_answer_ratio',
        'q_frac_count', 'q_number_count', 'q_operator_count',
        'q_parenthesis_count', 'q_decimal_count', 'q_variable_count',
        'mc_frac_count', 'mc_number_count', 'mc_operator_count',
        'mc_parenthesis_count', 'mc_decimal_count', 'mc_variable_count',
        'exp_frac_count', 'exp_number_count', 'exp_operator_count',
        'exp_parenthesis_count', 'exp_decimal_count', 'exp_variable_count'
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

    # 複数モデルの設定を取得
    model_configs = get_model_configs(n_classes)

    # 各モデルの予測を保存
    nsplit = 5
    skf = StratifiedKFold(n_splits=nsplit, shuffle=True, random_state=42)

    # 全モデルの予測を保存
    all_model_oof_preds = []
    all_model_test_preds = []
    model_scores = {}

    # 各モデルで学習
    print("\nTraining multiple XGBoost models...")
    for model_idx, config in enumerate(model_configs):
        print(f"\n{'='*60}")
        print(f"Training Model {model_idx + 1}/{len(model_configs)}: {config['name']}")
        print(f"{'='*60}")

        # モデルごとの予測を初期化
        model_oof_preds = np.zeros((len(train), n_classes))
        model_test_preds = np.zeros((len(test), n_classes))

        # 交差検証
        for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train_all, train['target_encoded'])):
            print(f"\nModel {model_idx + 1}, Fold {fold + 1}/{nsplit}")

            X_train_fold = X_train_all[train_idx]
            y_train_fold = train['target_encoded'].iloc[train_idx]
            X_valid_fold = X_train_all[valid_idx]
            y_valid_fold = train['target_encoded'].iloc[valid_idx]

            # DMatrix作成
            dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold, nthread=-1)
            dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold, nthread=-1)

            # モデル訓練
            model = xgb.train(
                config['params'],
                dtrain,
                num_boost_round=config['num_boost_round'],
                evals=[(dvalid, 'valid')],
                early_stopping_rounds=config['early_stopping_rounds'],
                verbose_eval=100
            )

            print(f"Best iteration: {model.best_iteration}")

            # 予測
            model_oof_preds[valid_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))
            model_test_preds += model.predict(xgb.DMatrix(X_test_all, nthread=-1), iteration_range=(0, model.best_iteration)) / nsplit

        # モデルの評価
        oof_pred_labels = np.argmax(model_oof_preds, axis=1)
        accuracy = np.mean(train['target_encoded'] == oof_pred_labels)
        f1 = sklearn.metrics.f1_score(train['target_encoded'], oof_pred_labels, average='weighted')

        # MAP@3の計算
        top3_indices = np.argsort(-model_oof_preds, axis=1)[:, :3]
        predictions = []
        for indices in top3_indices:
            predictions.append([target_classes[i] for i in indices])

        map_score = map3(train['target_cat'].tolist(), predictions)

        print(f"\nModel {config['name']} Performance:")
        print(f"  Validation Accuracy: {accuracy:.4f}")
        print(f"  Validation F1-score: {f1:.4f}")
        print(f"  Validation MAP@3: {map_score:.4f}")

        # スコアを保存
        model_scores[config['name']] = {
            'accuracy': accuracy,
            'f1': f1,
            'map3': map_score
        }

        # 予測を保存
        all_model_oof_preds.append(model_oof_preds)
        all_model_test_preds.append(model_test_preds)

    # アンサンブル（重み付き平均）
    print("\n" + "="*60)
    print("Ensemble predictions...")
    print("="*60)

    # MAP@3スコアに基づいて重みを計算
    map_scores = [model_scores[config['name']]['map3'] for config in model_configs]
    weights = np.array(map_scores) / np.sum(map_scores)

    print("\nModel weights based on MAP@3:")
    for i, config in enumerate(model_configs):
        print(f"  {config['name']}: {weights[i]:.4f} (MAP@3: {map_scores[i]:.4f})")

    # 重み付き平均でアンサンブル
    ensemble_oof_preds = np.zeros((len(train), n_classes))
    ensemble_test_preds = np.zeros((len(test), n_classes))

    for i, weight in enumerate(weights):
        ensemble_oof_preds += weight * all_model_oof_preds[i]
        ensemble_test_preds += weight * all_model_test_preds[i]

    # アンサンブルモデルの評価
    ensemble_pred_labels = np.argmax(ensemble_oof_preds, axis=1)
    ensemble_accuracy = np.mean(train['target_encoded'] == ensemble_pred_labels)
    ensemble_f1 = sklearn.metrics.f1_score(train['target_encoded'], ensemble_pred_labels, average='weighted')

    # MAP@3の計算
    ensemble_top3_indices = np.argsort(-ensemble_oof_preds, axis=1)[:, :3]
    ensemble_predictions = []
    for indices in ensemble_top3_indices:
        ensemble_predictions.append([target_classes[i] for i in indices])

    ensemble_map_score = map3(train['target_cat'].tolist(), ensemble_predictions)

    print(f"\nEnsemble Model Performance:")
    print(f"  Validation Accuracy: {ensemble_accuracy:.4f}")
    print(f"  Validation F1-score: {ensemble_f1:.4f}")
    print(f"  Validation MAP@3: {ensemble_map_score:.4f}")

    # 各モデルとの比較
    print("\nPerformance comparison:")
    print(f"{'Model':<20} {'Accuracy':<10} {'F1':<10} {'MAP@3':<10}")
    print("-" * 50)
    for config in model_configs:
        scores = model_scores[config['name']]
        print(f"{config['name']:<20} {scores['accuracy']:<10.4f} {scores['f1']:<10.4f} {scores['map3']:<10.4f}")
    print("-" * 50)
    print(f"{'Ensemble':<20} {ensemble_accuracy:<10.4f} {ensemble_f1:<10.4f} {ensemble_map_score:<10.4f}")

    # テスト予測
    test_top3_indices = np.argsort(-ensemble_test_preds, axis=1)[:, :3]
    test_predictions = []
    for indices in test_top3_indices:
        pred = [target_classes[i] for i in indices]
        test_predictions.append(' '.join(pred))

    # 提出ファイル作成
    submission = pd.read_csv(
        "/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv"
    )
    submission['Category:Misconception'] = test_predictions
    submission.to_csv("submission_ensemble.csv", index=False)
    print("\nSubmission file created: submission_ensemble.csv")
