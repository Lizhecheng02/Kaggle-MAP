# -*- coding: utf-8 -*-
"""
optimized_improve3_optuna.py

Map-Charting Student Math Misunderstandings
GPU使用率最大化版 + Optuna最適化 + メモリ最適化版
"""

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics
import re
from multiprocessing import Pool, cpu_count
import xgboost as xgb
from scipy import sparse
import warnings
import json
from datetime import datetime
import gc
warnings.filterwarnings('ignore')

# GPU設定
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# cuFileを無効化
os.environ['CUFILE_ENV_PATH'] = '/dev/null'
os.environ['CUFILE_DISABLE'] = '1'

# Optuna設定
OPTUNA_CONFIG = {
    'n_trials': 50,  # 試行回数
    'n_folds_optuna': 3,  # 最適化時のfold数（高速化のため）
    'n_folds_final': 5,  # 最終訓練時のfold数
    'optimize': True,  # Falseにすると最適化をスキップ
}

# メモリ最適化設定
MEMORY_CONFIG = {
    'max_features_limit': 5000,  # TF-IDF特徴量の上限を削減
    'batch_prediction': True,  # バッチ予測を有効化
    'prediction_batch_size': 1000,  # 予測時のバッチサイズ
    'sparse_output': True,  # スパース行列の使用を維持
    'gc_interval': 10,  # ガベージコレクションの頻度
}

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
    """バッチで特徴量抽出"""
    df_features = df.copy()
    
    # 基本的な長さ特徴
    df_features['mc_answer_len'] = df['MC_Answer'].astype(str).str.len()
    df_features['explanation_len'] = df['StudentExplanation'].astype(str).str.len()
    df_features['question_len'] = df['QuestionText'].astype(str).str.len()

    # 比率特徴
    df_features['explanation_to_question_ratio'] = df_features['explanation_len'] / (df_features['question_len'] + 1)
    df_features['answer_to_question_ratio'] = df_features['mc_answer_len'] / (df_features['question_len'] + 1)
    df_features['explanation_to_answer_ratio'] = df_features['explanation_len'] / (df_features['mc_answer_len'] + 1)

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

def create_features_with_params(train, test, tfidf_params):
    """パラメータ指定でTF-IDF特徴量作成"""
    
    # メモリ最適化：max_featuresを制限
    max_features = min(tfidf_params['max_features'], MEMORY_CONFIG['max_features_limit'])
    
    tfidf = TfidfVectorizer(
        ngram_range=(1, tfidf_params['ngram_max']),
        max_df=tfidf_params['max_df'],
        min_df=tfidf_params['min_df'],
        max_features=max_features,
        token_pattern=r'\b\w+\b',
        sublinear_tf=True,
        dtype=np.float32  # float32を使用してメモリ削減
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
    X_numeric = train[numeric_features].fillna(0).astype(np.float32).values
    X_numeric_test = test[numeric_features].fillna(0).astype(np.float32).values

    # スパース行列と数値特徴を結合
    X_train_all = sparse.hstack([
        train_tfidf,
        sparse.csr_matrix(X_numeric)
    ])
    X_test_all = sparse.hstack([
        test_tfidf,
        sparse.csr_matrix(X_numeric_test)
    ])

    return X_train_all, X_test_all

def predict_batch(model, dmatrix, batch_size=1000):
    """バッチ予測でメモリ使用量を削減"""
    n_samples = dmatrix.num_row()
    predictions = []
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_dmatrix = dmatrix.slice(list(range(i, end_idx)))
        batch_pred = model.predict(batch_dmatrix, iteration_range=(0, model.best_iteration))
        predictions.append(batch_pred)
        
        # メモリ解放
        del batch_dmatrix
        if i % (batch_size * 5) == 0:
            gc.collect()
    
    return np.vstack(predictions)

def objective(trial, train, test, n_classes, target_classes):
    """Optuna目的関数"""
    # TF-IDFパラメータ
    tfidf_params = {
        'ngram_max': trial.suggest_int('tfidf_ngram_max', 2, 3),  # 最大値を削減
        'max_df': trial.suggest_float('tfidf_max_df', 0.8, 0.95),
        'min_df': trial.suggest_int('tfidf_min_df', 2, 10),  # 最小値を増加
        'max_features': trial.suggest_int('tfidf_max_features', 2000, 5000, step=500),  # 範囲を削減
    }
    
    # XGBoostパラメータ
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
        'max_depth': trial.suggest_int('max_depth', 6, 12),  # 深さを削減
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'alpha': trial.suggest_float('alpha', 0.0, 1.0),
        'lambda': trial.suggest_float('lambda', 0.5, 3.0),
        'random_state': 42,
        # GPU最適化パラメータ（メモリ効率化）
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'gpu_id': 0,
        'max_bin': trial.suggest_categorical('max_bin', [64, 128, 256]),  # 最大値を削減
        'grow_policy': 'lossguide',  # メモリ効率的な成長ポリシー
        'single_precision_histogram': True,
        'updater': 'grow_gpu_hist',
        'sampling_method': 'gradient_based',
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 5),
        # メモリ最適化パラメータ
        'max_leaves': trial.suggest_int('max_leaves', 31, 127),  # 葉の数を制限
    }
    
    # 特徴量作成
    try:
        X_train_all, X_test_all = create_features_with_params(train, test, tfidf_params)
    except Exception as e:
        print(f"Feature creation failed: {e}")
        return float('inf')
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=OPTUNA_CONFIG['n_folds_optuna'], shuffle=True, random_state=42)
    
    oof_preds = np.zeros((len(train), n_classes))
    fold_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train_all, train['target_encoded'])):
        try:
            X_train_fold = X_train_all[train_idx]
            y_train_fold = train['target_encoded'].iloc[train_idx]
            X_valid_fold = X_train_all[valid_idx]
            y_valid_fold = train['target_encoded'].iloc[valid_idx]
            
            # DMatrix作成
            dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold, nthread=-1)
            dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold, nthread=-1)
            
            # モデル訓練
            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=1000,
                evals=[(dvalid, 'valid')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            # 予測（メモリ効率化のためバッチ処理）
            if MEMORY_CONFIG['batch_prediction']:
                oof_preds[valid_idx] = predict_batch(model, dvalid, MEMORY_CONFIG['prediction_batch_size'])
            else:
                oof_preds[valid_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))
            
            # メモリ解放
            del dtrain, dvalid, model
            gc.collect()
            
            # Pruning（早期終了）
            trial.report(-1, fold)  # ダミー値を報告
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        except Exception as e:
            print(f"Error in fold {fold}: {e}")
            # メモリエラーの場合は大きなペナルティ
            if "out of memory" in str(e).lower():
                return float('inf')
            raise e
    
    # MAP@3の計算
    top3_indices = np.argsort(-oof_preds, axis=1)[:, :3]
    predictions = []
    for indices in top3_indices:
        predictions.append([target_classes[i] for i in indices])
    
    map_score = map3(train['target_cat'].tolist(), predictions)
    
    # メモリ解放
    gc.collect()
    
    return -map_score  # 最大化のため負の値を返す

if __name__ == '__main__':
    print('Starting GPU-optimized model training with Optuna (Memory Optimized)...')

    # GPU使用状況の確認
    try:
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
    except Exception as e:
        print(f"GPU info not available: {e}")

    # データ読み込み（通常のpandas使用）
    print("Loading data with pandas...")
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

    # 特徴量作成
    print("Extracting features...")
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

    # Optuna最適化
    if OPTUNA_CONFIG['optimize']:
        print("\nStarting Optuna optimization...")
        
        # Optuna設定
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner
        )
        
        # 最適化実行
        study.optimize(
            lambda trial: objective(trial, train, test, n_classes, target_classes),
            n_trials=OPTUNA_CONFIG['n_trials'],
            show_progress_bar=True,
            catch=(Exception,)  # すべての例外をキャッチ
        )
        
        # 最適パラメータ
        best_params = study.best_params
        print(f"\nBest MAP@3: {-study.best_value:.4f}")
        print("Best parameters:")
        for key, value in sorted(best_params.items()):
            print(f"  {key}: {value}")
        
        # 最適パラメータを保存
        with open('best_params_optuna_memory.json', 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_map3': -study.best_value,
                'n_trials': len(study.trials),
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=4)
    else:
        # 既存の最適パラメータを読み込み
        print("\nLoading existing best parameters...")
        with open('best_params_optuna_memory.json', 'r') as f:
            saved_params = json.load(f)
        best_params = saved_params['best_params']
        print(f"Loaded best MAP@3: {saved_params['best_map3']:.4f}")

    # 最適パラメータでTF-IDF特徴量作成
    print("\nCreating final features with best parameters...")
    tfidf_params = {
        'ngram_max': best_params.get('tfidf_ngram_max', 2),
        'max_df': best_params.get('tfidf_max_df', 0.9),
        'min_df': best_params.get('tfidf_min_df', 3),
        'max_features': best_params.get('tfidf_max_features', 3000),
    }

    # 最終的なXGBoostパラメータ
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
        'max_depth': best_params.get('max_depth', 8),
        'learning_rate': best_params.get('learning_rate', 0.03),
        'subsample': best_params.get('subsample', 0.8),
        'colsample_bytree': best_params.get('colsample_bytree', 0.8),
        'min_child_weight': best_params.get('min_child_weight', 2),
        'gamma': best_params.get('gamma', 0.1),
        'alpha': best_params.get('alpha', 0.1),
        'lambda': best_params.get('lambda', 1.5),
        'random_state': 42,
        # GPU最適化パラメータ
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'gpu_id': 0,
        'max_bin': best_params.get('max_bin', 128),
        'grow_policy': 'lossguide',
        'single_precision_histogram': True,
        'updater': 'grow_gpu_hist',
        'sampling_method': 'gradient_based',
        'max_delta_step': best_params.get('max_delta_step', 1),
        'max_leaves': best_params.get('max_leaves', 63),
    }

    # Optuna最適化結果から上位5個の設定を出力
    if OPTUNA_CONFIG['optimize']:
        print("\n" + "="*50)
        print("TOP 5 BEST CONFIGURATIONS")
        print("="*50)
        
        # 試行結果をMAP@3スコアでソート（降順）
        sorted_trials = sorted(study.trials, key=lambda x: x.value)[:5]
        
        for i, trial in enumerate(sorted_trials):
            print(f"\n--- Configuration #{i+1} ---")
            print(f"MAP@3 Score: {-trial.value:.4f}")
            print("Parameters:")
            
            # パラメータを整理して表示
            params = trial.params
            
            # TF-IDFパラメータ
            print("\nTF-IDF Parameters:")
            print(f"  ngram_max: {params.get('tfidf_ngram_max')}")
            print(f"  max_df: {params.get('tfidf_max_df')}")
            print(f"  min_df: {params.get('tfidf_min_df')}")
            print(f"  max_features: {params.get('tfidf_max_features')}")
            
            # XGBoostパラメータ
            print("\nXGBoost Parameters:")
            print(f"  max_depth: {params.get('max_depth')}")
            print(f"  learning_rate: {params.get('learning_rate'):.4f}")
            print(f"  subsample: {params.get('subsample'):.4f}")
            print(f"  colsample_bytree: {params.get('colsample_bytree'):.4f}")
            print(f"  min_child_weight: {params.get('min_child_weight')}")
            print(f"  gamma: {params.get('gamma'):.4f}")
            print(f"  alpha: {params.get('alpha'):.4f}")
            print(f"  lambda: {params.get('lambda'):.4f}")
            print(f"  max_bin: {params.get('max_bin')}")
            print(f"  max_delta_step: {params.get('max_delta_step')}")
            print(f"  max_leaves: {params.get('max_leaves')}")
        
        # 上位5個の設定をJSONファイルに保存
        top5_configs = []
        for i, trial in enumerate(sorted_trials):
            config = {
                'rank': i + 1,
                'map3_score': -trial.value,
                'params': trial.params
            }
            top5_configs.append(config)
        
        with open('top5_configurations.json', 'w') as f:
            json.dump({
                'top5_configurations': top5_configs,
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=4)
        
        print(f"\n\nTop 5 configurations saved to: top5_configurations.json")
        
    else:
        print("\nOptimization was skipped. Run with OPTUNA_CONFIG['optimize'] = True to get configurations.")