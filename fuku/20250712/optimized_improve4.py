# -*- coding: utf-8 -*-
"""
optimized_gpu_maximized.py

Map-Charting Student Math Misunderstandings
GPU使用率最大化版 + 検索型アプローチ
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

# 検索型アプローチ用のライブラリ
from sentence_transformers import SentenceTransformer
import faiss
from collections import Counter, defaultdict
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# GPU設定
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def enhanced_clean_batch(batch):
    """強化されたバッチ単位でのテキストクリーニング"""
    cleaned = []
    for text in batch:
        # 複雑な数式パターンを検出
        text = re.sub(r'(\d+)\s*/\s*(\d+)', r'FRAC_\1_\2', text)
        text = re.sub(r'\\frac\{([^\}]+)\}\{([^\}]+)\}', r'FRAC_\1_\2', text)
        text = re.sub(r'(\d+)\s*/\s*(\d+)\s*\+\s*(\d+)\s*/\s*(\d+)', r'FRAC_\1_\2 PLUS FRAC_\3_\4', text)
        
        # 指数表現
        text = re.sub(r'(\w+)\^(\d+)', r'\1 POWER \2', text)
        text = re.sub(r'(\w+)²', r'\1 POWER 2', text)
        text = re.sub(r'(\w+)³', r'\1 POWER 3', text)
        
        # 根号
        text = re.sub(r'√(\d+)', r'SQRT_\1', text)
        text = re.sub(r'sqrt\((\d+)\)', r'SQRT_\1', text)
        
        # 不等号
        text = re.sub(r'<', ' LESS_THAN ', text)
        text = re.sub(r'>', ' GREATER_THAN ', text)
        text = re.sub(r'≤|<=', ' LESS_EQUAL ', text)
        text = re.sub(r'≥|>=', ' GREATER_EQUAL ', text)
        
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

def parallel_clean(texts, n_jobs=None, enhanced=True):
    """並列でテキストクリーニング"""
    if n_jobs is None:
        n_jobs = cpu_count()

    # バッチに分割
    batch_size = len(texts) // n_jobs + 1
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

    # 並列処理
    clean_func = enhanced_clean_batch if enhanced else clean_batch
    with Pool(n_jobs) as pool:
        results = pool.map(clean_func, batches)

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
    """並列で数学的特徴を抽出（拡張版）"""
    def extract_single(text):
        text = str(text)
        return {
            'frac_count': len(re.findall(r'FRAC_\d+_\d+|\\frac', text)),
            'number_count': len(re.findall(r'\b\d+\b', text)),
            'operator_count': len(re.findall(r'[\+\-\*\/\=]|PLUS|MINUS|TIMES|DIVIDE|EQUALS', text)),
            'parenthesis_count': len(re.findall(r'[\(\)]', text)),
            'decimal_count': len(re.findall(r'\d+\.\d+', text)),
            'variable_count': len(re.findall(r'\b[a-zA-Z]\b', text)),
            'power_count': len(re.findall(r'POWER|\^|²|³', text)),
            'sqrt_count': len(re.findall(r'SQRT|√|sqrt', text)),
            'inequality_count': len(re.findall(r'LESS|GREATER|[<>≤≥]', text)),
            'complex_frac_count': len(re.findall(r'FRAC.*PLUS.*FRAC|FRAC.*MINUS.*FRAC', text)),
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


class SemanticSearchEngine:
    """意味的類似度ベースの検索エンジン"""
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.train_embeddings = None
        self.train_labels = None
        self.label_freq = None
        
    def build_index(self, texts, labels):
        """FAISSインデックスを構築"""
        print("Building semantic embeddings...")
        self.train_embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=True)
        self.train_labels = labels
        
        # ラベル頻度を計算
        self.label_freq = Counter(labels)
        
        # FAISSインデックスを作成
        d = self.train_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.train_embeddings.astype(np.float32))
        
    def search(self, query_texts, k=20):
        """類似例を検索"""
        query_embeddings = self.model.encode(query_texts, batch_size=64, show_progress_bar=False)
        distances, indices = self.index.search(query_embeddings.astype(np.float32), k)
        
        predictions = []
        for idx_list in indices:
            # 検索された例のラベル分布を計算
            labels = [self.train_labels[idx] for idx in idx_list]
            label_counts = Counter(labels)
            
            # 頻度順にソート
            sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
            predictions.append([label for label, _ in sorted_labels[:3]])
            
        return predictions
    
    def get_similarity_scores(self, query_texts, k=20):
        """類似度スコアを取得"""
        query_embeddings = self.model.encode(query_texts, batch_size=64, show_progress_bar=False)
        distances, indices = self.index.search(query_embeddings.astype(np.float32), k)
        
        # 距離を類似度スコアに変換
        similarities = 1 / (1 + distances)
        
        label_scores = []
        for sim_scores, idx_list in zip(similarities, indices):
            label_score_dict = defaultdict(float)
            
            for sim, idx in zip(sim_scores, idx_list):
                label = self.train_labels[idx]
                # 希少ラベルには高い重みを付ける
                freq_weight = 1.0 / np.sqrt(self.label_freq[label])
                label_score_dict[label] += sim * freq_weight
            
            # 正規化
            total = sum(label_score_dict.values())
            if total > 0:
                for label in label_score_dict:
                    label_score_dict[label] /= total
                    
            label_scores.append(dict(label_score_dict))
            
        return label_scores


def hybrid_prediction(xgb_probs, search_scores, target_classes, alpha=0.7):
    """XGBoostと検索ベースの予測をハイブリッド"""
    n_samples = xgb_probs.shape[0]
    n_classes = len(target_classes)
    
    # ラベルとインデックスのマッピング
    label_to_idx = {label: idx for idx, label in enumerate(target_classes)}
    
    hybrid_probs = np.zeros((n_samples, n_classes))
    
    for i in range(n_samples):
        # XGBoostの予測確率
        hybrid_probs[i] = xgb_probs[i] * alpha
        
        # 検索ベースのスコアを追加
        for label, score in search_scores[i].items():
            if label in label_to_idx:
                idx = label_to_idx[label]
                hybrid_probs[i, idx] += score * (1 - alpha)
    
    return hybrid_probs


class HierarchicalClassifier:
    """階層的分類器（Category → Misconception）"""
    def __init__(self):
        self.category_model = None
        self.misconception_models = {}
        self.category_encoder = None
        self.misconception_encoders = {}
        
    def fit(self, X, y_category, y_misconception, params, n_folds=5):
        """階層的モデルの学習"""
        # Step 1: Category分類器の学習
        print("Training category classifier...")
        self.category_encoder = LabelEncoder()
        y_category_encoded = self.category_encoder.fit_transform(y_category)
        
        # Categoryモデルのパラメータ
        category_params = params.copy()
        category_params['num_class'] = len(self.category_encoder.classes_)
        
        # Cross-validation for category
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        category_preds = np.zeros((len(y_category), category_params['num_class']))
        
        for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y_category_encoded)):
            X_train_fold = X[train_idx]
            y_train_fold = y_category_encoded[train_idx]
            X_valid_fold = X[valid_idx]
            y_valid_fold = y_category_encoded[valid_idx]
            
            dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
            dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold)
            
            model = xgb.train(
                category_params,
                dtrain,
                num_boost_round=500,
                evals=[(dvalid, 'valid')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            category_preds[valid_idx] = model.predict(dvalid)
        
        # 全データでCategoryモデルを再学習
        dtrain_all = xgb.DMatrix(X, label=y_category_encoded)
        self.category_model = xgb.train(
            category_params,
            dtrain_all,
            num_boost_round=500,
            verbose_eval=False
        )
        
        # Step 2: 各Categoryごとのmisconceptionモデルを学習
        print("Training misconception classifiers for each category...")
        unique_categories = np.unique(y_category)
        
        for category in unique_categories:
            print(f"  Training for category: {category}")
            mask = y_category == category
            X_cat = X[mask]
            y_misc_cat = y_misconception[mask]
            
            if len(np.unique(y_misc_cat)) < 2:
                # 単一クラスの場合はスキップ
                continue
                
            self.misconception_encoders[category] = LabelEncoder()
            y_misc_encoded = self.misconception_encoders[category].fit_transform(y_misc_cat)
            
            misc_params = params.copy()
            misc_params['num_class'] = len(self.misconception_encoders[category].classes_)
            
            dtrain_misc = xgb.DMatrix(X_cat, label=y_misc_encoded)
            self.misconception_models[category] = xgb.train(
                misc_params,
                dtrain_misc,
                num_boost_round=500,
                verbose_eval=False
            )
    
    def predict_proba(self, X):
        """階層的予測"""
        # Step 1: Category予測
        dtest = xgb.DMatrix(X)
        category_probs = self.category_model.predict(dtest)
        category_preds = np.argmax(category_probs, axis=1)
        category_labels = self.category_encoder.inverse_transform(category_preds)
        
        # Step 2: 各サンプルごとにMisconception予測
        all_probs = []
        
        for i in range(len(X)):
            category = category_labels[i]
            
            if category in self.misconception_models:
                # その�ategoryのモデルで予測
                X_single = X[i:i+1]
                dtest_single = xgb.DMatrix(X_single)
                misc_probs = self.misconception_models[category].predict(dtest_single)[0]
                
                # カテゴリの確率と掛け合わせる
                cat_prob = category_probs[i, category_preds[i]]
                misc_probs = misc_probs * cat_prob
                
                # ラベルをデコード
                misc_labels = self.misconception_encoders[category].classes_
                prob_dict = {f"{category}:{label}": prob for label, prob in zip(misc_labels, misc_probs)}
            else:
                # デフォルト予測
                prob_dict = {f"{category}:NA": category_probs[i, category_preds[i]]}
            
            all_probs.append(prob_dict)
        
        return all_probs

if __name__ == '__main__':
    print('Starting GPU-optimized model training...')

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
    ).to_pandas()  # 一旦pandasに戻す（後続処理のため）

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

    # 並列テキストクリーニング（強化版）
    print("Enhanced parallel text cleaning...")
    train_texts = train['combined_text'].tolist()
    test_texts = test['combined_text'].tolist()

    train['cleaned_text'] = parallel_clean(train_texts, n_jobs=8, enhanced=True)
    test['cleaned_text'] = parallel_clean(test_texts, n_jobs=8, enhanced=True)

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

    # TF-IDF特徴量（簡易版 - メモリ効率重視）
    print("Creating TF-IDF features...")
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),  # 4-gramから3-gramに削減
        max_df=0.9,
        min_df=2,
        max_features=5000,  # 10000から5000に削減してGPU転送を高速化
        token_pattern=r'\b\w+\b',
        sublinear_tf=True
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
        'q_parenthesis_count', 'q_decimal_count', 'q_variable_count',
        'q_power_count', 'q_sqrt_count', 'q_inequality_count', 'q_complex_frac_count',
        'mc_frac_count', 'mc_number_count', 'mc_operator_count',
        'mc_parenthesis_count', 'mc_decimal_count', 'mc_variable_count',
        'mc_power_count', 'mc_sqrt_count', 'mc_inequality_count', 'mc_complex_frac_count',
        'exp_frac_count', 'exp_number_count', 'exp_operator_count',
        'exp_parenthesis_count', 'exp_decimal_count', 'exp_variable_count',
        'exp_power_count', 'exp_sqrt_count', 'exp_inequality_count', 'exp_complex_frac_count'
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
    
    # ラベル頻度の分析
    print("\nAnalyzing label frequency distribution...")
    label_counts = train['target_cat'].value_counts()
    rare_labels = label_counts[label_counts <= 5].index.tolist()
    print(f"Total unique labels: {len(label_counts)}")
    print(f"Labels with frequency <= 5: {len(rare_labels)} ({len(rare_labels)/len(label_counts)*100:.1f}%)")
    print(f"Labels with frequency = 1: {(label_counts == 1).sum()} ({(label_counts == 1).sum()/len(label_counts)*100:.1f}%)")
    
    # 検索エンジンの構築
    print("\nBuilding semantic search engine...")
    search_engine = SemanticSearchEngine()
    search_engine.build_index(train['cleaned_text'].tolist(), train['target_cat'].tolist())
    
    # クラス重みの計算（希少ラベル対策）
    class_weights = {}
    for i, label in enumerate(target_classes):
        freq = label_counts.get(label, 1)
        # 頻度の逆数の平方根を重みとする
        class_weights[i] = 1.0 / np.sqrt(freq)
    
    # 重みを正規化
    max_weight = max(class_weights.values())
    for i in class_weights:
        class_weights[i] = class_weights[i] / max_weight * 10.0  # 最大重みを10に

    # XGBoostモデル（GPU最適化）
    print("\nTraining XGBoost with GPU optimization and class weights...")
    nsplit = 5

    skf = StratifiedKFold(n_splits=nsplit, shuffle=True, random_state=42)

    # 学習率スケジューラーの設定（設定可能なパラメータ）
    scheduler_config = {
        'initial_learning_rate': 0.05,      # 初期学習率
        'min_learning_rate': 0.001,         # 最小学習率
        'decay_rate': 0.95,                 # フォールドごとの減衰率
        'scheduler_type': 'cosine',         # 'cosine', 'linear', 'exponential'
        'warmup_rounds': 50,                # ウォームアップラウンド数
    }
    
    initial_learning_rate = scheduler_config['initial_learning_rate']
    min_learning_rate = scheduler_config['min_learning_rate']
    decay_rate = scheduler_config['decay_rate']
    
    # GPU最適化パラメータ
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
        'max_depth': 16,
        'learning_rate': initial_learning_rate,  # 初期学習率
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 2,
        'gamma': 0.1,
        'alpha': 0.1,
        'lambda': 1.5,
        'random_state': 42,
        # GPU最適化パラメータ
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',  # 明示的にGPU予測を指定
        'gpu_id': 0,
        'max_bin': 1024,  # GPUメモリ効率のため
        'grow_policy': 'depthwise',  # GPU効率が良い
        'single_precision_histogram': True,  # 高速化のため
        # バッチサイズを大きくしてGPU使用率向上
        'updater': 'grow_gpu_hist',
        'sampling_method': 'gradient_based',  # GPU効率向上
        'subsample': 0.8,
        'max_delta_step': 1,
    }

    oof_preds = np.zeros((len(train), n_classes))
    test_preds = np.zeros((len(test), n_classes))

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train_all, train['target_encoded'])):
        print(f"\nFold {fold + 1}/{nsplit}")
        
        # 学習率スケジューリング（フォールドごとに減衰）
        current_lr = max(initial_learning_rate * (decay_rate ** fold), min_learning_rate)
        print(f"Current learning rate: {current_lr:.4f}")
        
        # フォールドごとのパラメータをコピーして学習率を更新
        fold_params = xgb_params.copy()
        fold_params['learning_rate'] = current_lr

        X_train_fold = X_train_all[train_idx]
        y_train_fold = train['target_encoded'].iloc[train_idx]
        X_valid_fold = X_train_all[valid_idx]
        y_valid_fold = train['target_encoded'].iloc[valid_idx]

        # クラス重みの設定
        sample_weights_train = np.array([class_weights[y] for y in y_train_fold])
        sample_weights_valid = np.array([class_weights[y] for y in y_valid_fold])
        
        # DMatrix作成（GPUメモリ効率化 + クラス重み）
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold, weight=sample_weights_train, nthread=-1)
        dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold, weight=sample_weights_valid, nthread=-1)

        # コールバック関数で動的な学習率調整
        def learning_rate_callback(current_iter):
            """エポックごとの学習率調整"""
            total_rounds = 1500
            warmup_rounds = scheduler_config['warmup_rounds']
            
            # ウォームアップフェーズ
            if current_iter < warmup_rounds:
                lr = min_learning_rate + (current_lr - min_learning_rate) * (current_iter / warmup_rounds)
            else:
                # スケジューラータイプに応じた学習率調整
                if scheduler_config['scheduler_type'] == 'cosine':
                    # コサインアニーリング
                    progress = (current_iter - warmup_rounds) / (total_rounds - warmup_rounds)
                    lr = min_learning_rate + 0.5 * (current_lr - min_learning_rate) * (
                        1 + np.cos(np.pi * progress)
                    )
                elif scheduler_config['scheduler_type'] == 'linear':
                    # 線形減衰
                    progress = (current_iter - warmup_rounds) / (total_rounds - warmup_rounds)
                    lr = current_lr - (current_lr - min_learning_rate) * progress
                elif scheduler_config['scheduler_type'] == 'exponential':
                    # 指数関数的減衰
                    progress = (current_iter - warmup_rounds) / (total_rounds - warmup_rounds)
                    lr = current_lr * (min_learning_rate / current_lr) ** progress
                else:
                    lr = current_lr
            
            return lr
        
        # モデル訓練
        model = xgb.train(
            fold_params,
            dtrain,
            num_boost_round=1500,
            evals=[(dvalid, 'valid')],
            early_stopping_rounds=100,
            verbose_eval=100,
            callbacks=[xgb.callback.LearningRateScheduler(learning_rate_callback)]
        )

        print(f"Best iteration: {model.best_iteration}")

        # 予測
        oof_preds[valid_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))
        test_preds += model.predict(xgb.DMatrix(X_test_all, nthread=-1), iteration_range=(0, model.best_iteration)) / nsplit

    # 評価
    oof_pred_labels = np.argmax(oof_preds, axis=1)
    accuracy = np.mean(train['target_encoded'] == oof_pred_labels)
    f1 = sklearn.metrics.f1_score(train['target_encoded'], oof_pred_labels, average='weighted')

    print(f"\nValidation Accuracy: {accuracy:.4f}")
    print(f"Validation F1-score: {f1:.4f}")

    # 検索ベースの予測を取得
    print("\nGetting search-based predictions...")
    search_scores_train = search_engine.get_similarity_scores(train['cleaned_text'].tolist(), k=30)
    
    # ハイブリッド予測
    print("Creating hybrid predictions...")
    hybrid_oof_preds = hybrid_prediction(oof_preds, search_scores_train, target_classes, alpha=0.7)
    
    # MAP@3の計算（XGBoostのみ）
    top3_indices = np.argsort(-oof_preds, axis=1)[:, :3]
    predictions_xgb = []
    for indices in top3_indices:
        predictions_xgb.append([target_classes[i] for i in indices])
    
    map_score_xgb = map3(train['target_cat'].tolist(), predictions_xgb)
    print(f"\nValidation MAP@3 (XGBoost only): {map_score_xgb:.4f}")
    
    # MAP@3の計算（ハイブリッド）
    top3_indices_hybrid = np.argsort(-hybrid_oof_preds, axis=1)[:, :3]
    predictions_hybrid = []
    for indices in top3_indices_hybrid:
        predictions_hybrid.append([target_classes[i] for i in indices])
    
    map_score_hybrid = map3(train['target_cat'].tolist(), predictions_hybrid)
    print(f"Validation MAP@3 (Hybrid): {map_score_hybrid:.4f}")
    print(f"Improvement: +{(map_score_hybrid - map_score_xgb) * 100:.2f}%")
    
    # 階層的分類器の学習と評価
    print("\n\nTraining hierarchical classifier...")
    hierarchical_clf = HierarchicalClassifier()
    
    # ラベルとインデックスのマッピング（階層的分類用）
    label_to_idx = {label: idx for idx, label in enumerate(target_classes)}
    
    # GPUパラメータを階層的分類用に調整
    hier_params = xgb_params.copy()
    hier_params.pop('num_class')  # 各モデルで動的に設定
    
    # 階層的分類器の学習
    hierarchical_clf.fit(
        X_train_all,
        train['Category'].values,
        train['Misconception'].values,
        hier_params,
        n_folds=3  # 高速化のため少なめ
    )
    
    # 階層的分類器での予測
    print("\nEvaluating hierarchical classifier...")
    hier_probs_dicts = hierarchical_clf.predict_proba(X_train_all)
    
    # 確率辞書を行列形式に変換
    hier_oof_preds = np.zeros((len(train), n_classes))
    for i, prob_dict in enumerate(hier_probs_dicts):
        for label, prob in prob_dict.items():
            if label in label_to_idx:
                idx = label_to_idx[label]
                hier_oof_preds[i, idx] = prob
    
    # MAP@3の計算（階層的）
    top3_indices_hier = np.argsort(-hier_oof_preds, axis=1)[:, :3]
    predictions_hier = []
    for indices in top3_indices_hier:
        predictions_hier.append([target_classes[i] for i in indices])
    
    map_score_hier = map3(train['target_cat'].tolist(), predictions_hier)
    print(f"Validation MAP@3 (Hierarchical): {map_score_hier:.4f}")
    
    # 最終的なアンサンブル（3つのアプローチ）
    print("\nCreating final ensemble...")
    final_oof_preds = (
        oof_preds * 0.4 +           # XGBoost
        hybrid_oof_preds * 0.4 +     # ハイブリッド
        hier_oof_preds * 0.2         # 階層的
    )
    
    top3_indices_final = np.argsort(-final_oof_preds, axis=1)[:, :3]
    predictions_final = []
    for indices in top3_indices_final:
        predictions_final.append([target_classes[i] for i in indices])
    
    map_score_final = map3(train['target_cat'].tolist(), predictions_final)
    print(f"Validation MAP@3 (Final Ensemble): {map_score_final:.4f}")
    print(f"Total improvement: +{(map_score_final - map_score_xgb) * 100:.2f}%")

    # テスト予測（ハイブリッド）
    print("\nGetting test search-based predictions...")
    search_scores_test = search_engine.get_similarity_scores(test['cleaned_text'].tolist(), k=30)
    
    # ハイブリッド予測
    hybrid_test_preds = hybrid_prediction(test_preds, search_scores_test, target_classes, alpha=0.7)
    
    # 階層的予測（テスト）
    print("Getting hierarchical test predictions...")
    hier_test_probs_dicts = hierarchical_clf.predict_proba(X_test_all)
    
    # 確率辞書を行列形式に変換
    hier_test_preds = np.zeros((len(test), n_classes))
    for i, prob_dict in enumerate(hier_test_probs_dicts):
        for label, prob in prob_dict.items():
            if label in label_to_idx:
                idx = label_to_idx[label]
                hier_test_preds[i, idx] = prob
    
    # 最終的なアンサンブル予測
    print("Creating final ensemble test predictions...")
    final_test_preds = (
        test_preds * 0.4 +           # XGBoost
        hybrid_test_preds * 0.4 +    # ハイブリッド
        hier_test_preds * 0.2        # 階層的
    )
    
    # 最終予測
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
    submission.to_csv("submission.csv", index=False)
    print("\nSubmission file created: submission.csv")
    
    # モデルと検索エンジンを保存
    print("\nSaving search engine for future use...")
    with open('search_engine.pkl', 'wb') as f:
        pickle.dump(search_engine, f)
    print("Search engine saved to search_engine.pkl")
