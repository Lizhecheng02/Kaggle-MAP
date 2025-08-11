# -*- coding: utf-8 -*-
"""
optimized_claude.py

MAP@3スコア向上のための最適化版（XGBoost単体）
- より高度な特徴量エンジニアリング
- 単一XGBoostモデル
"""

import numpy as np
import pandas as pd
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
import warnings
warnings.filterwarnings('ignore')

# NLTK WordNetデータをダウンロード
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

def advanced_clean(text):
    """テキストの高度なクリーニング"""
    # 数式パターンを特殊トークンに変換
    text = re.sub(r'(\d+)\s*/\s*(\d+)', r'FRAC_\1_\2', text)
    text = re.sub(r'\\frac\{([^\}]+)\}\{([^\}]+)\}', r'FRAC_\1_\2', text)
    text = re.sub(r'(\d+)/(\d+)', r'FRAC_\1_\2', text)
    
    # 演算子をトークン化
    text = re.sub(r'(\d+)\s*\+\s*(\d+)', r'\1 PLUS \2', text)
    text = re.sub(r'(\d+)\s*\-\s*(\d+)', r'\1 MINUS \2', text)
    text = re.sub(r'(\d+)\s*\*\s*(\d+)', r'\1 TIMES \2', text)
    text = re.sub(r'(\d+)\s*×\s*(\d+)', r'\1 TIMES \2', text)
    text = re.sub(r'(\d+)\s*÷\s*(\d+)', r'\1 DIVIDE \2', text)
    text = re.sub(r'=', ' EQUALS ', text)
    text = re.sub(r'>', ' GREATER ', text)
    text = re.sub(r'<', ' LESS ', text)
    text = re.sub(r'≥', ' GREATER_EQUAL ', text)
    text = re.sub(r'≤', ' LESS_EQUAL ', text)
    
    # 数学記号の処理
    text = re.sub(r'\^(\d+)', r' POWER_\1 ', text)
    text = re.sub(r'√', ' SQRT ', text)
    text = re.sub(r'π', ' PI ', text)
    
    # 基本的なクリーニング
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s_]', '', text)
    
    return text.strip().lower()

def extract_math_features(text):
    """数学的特徴を抽出"""
    features = {}
    
    # 分数の特徴
    features['frac_count'] = len(re.findall(r'FRAC_\d+_\d+|/|\\\frac', text))
    proper_fractions = re.findall(r'FRAC_(\d+)_(\d+)', text)
    features['proper_frac_count'] = sum(1 for n, d in proper_fractions if int(n) < int(d))
    features['improper_frac_count'] = sum(1 for n, d in proper_fractions if int(n) >= int(d))
    
    # 数値の特徴
    numbers = re.findall(r'\b\d+\.?\d*\b', text)
    features['number_count'] = len(numbers)
    if numbers:
        nums = [float(n) for n in numbers]
        features['max_number'] = max(nums)
        features['min_number'] = min(nums)
        features['number_range'] = max(nums) - min(nums)
        features['number_std'] = np.std(nums)
        features['has_large_numbers'] = int(any(n > 100 for n in nums))
        features['has_decimals'] = int(any('.' in n for n in numbers))
        features['unique_numbers'] = len(set(numbers))
    else:
        features['max_number'] = 0
        features['min_number'] = 0
        features['number_range'] = 0
        features['number_std'] = 0
        features['has_large_numbers'] = 0
        features['has_decimals'] = 0
        features['unique_numbers'] = 0
    
    # 演算子の特徴
    features['plus_count'] = len(re.findall(r'PLUS|\+', text))
    features['minus_count'] = len(re.findall(r'MINUS|\-', text))
    features['times_count'] = len(re.findall(r'TIMES|\*|×', text))
    features['divide_count'] = len(re.findall(r'DIVIDE|÷', text))
    features['equals_count'] = len(re.findall(r'EQUALS|=', text))
    features['total_operators'] = sum([features['plus_count'], features['minus_count'], 
                                      features['times_count'], features['divide_count']])
    
    # 括弧とその他の記号
    features['parenthesis_count'] = len(re.findall(r'[\(\)]', text))
    features['bracket_count'] = len(re.findall(r'[\[\]]', text))
    features['decimal_count'] = len(re.findall(r'\d+\.\d+', text))
    features['percent_count'] = len(re.findall(r'%|percent', text))
    
    # 変数の数
    features['variable_count'] = len(re.findall(r'\b[a-zA-Z]\b', text))
    features['has_equation'] = int('EQUALS' in text or '=' in text)
    
    # 数学用語の数
    math_terms = ['equation', 'solve', 'calculate', 'formula', 'variable', 'fraction', 
                  'decimal', 'percent', 'ratio', 'proportion', 'sum', 'difference', 
                  'product', 'quotient', 'remainder', 'factor', 'multiple', 'divisor',
                  'numerator', 'denominator', 'integer', 'whole', 'mixed', 'improper',
                  'simplify', 'reduce', 'equivalent', 'value', 'expression', 'term']
    features['math_term_count'] = sum(1 for term in math_terms if term in text.lower())
    
    return features

def extract_statistical_features(text):
    """統計的特徴を抽出"""
    features = {}
    
    # 単語の統計
    words = text.split()
    features['word_count'] = len(words)
    features['char_count'] = len(text)
    features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
    features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
    
    # 文の統計
    sentences = re.split(r'[.!?]+', text)
    features['sentence_count'] = len([s for s in sentences if s.strip()])
    features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0
    
    # 大文字の割合
    original_text = text
    features['uppercase_ratio'] = sum(1 for c in original_text if c.isupper()) / len(original_text) if original_text else 0
    
    # 句読点の特徴
    features['punctuation_count'] = sum(1 for c in text if c in string.punctuation)
    features['question_mark_count'] = text.count('?')
    features['exclamation_count'] = text.count('!')
    
    return features

def extract_error_patterns(text):
    """エラーパターンを抽出"""
    patterns = {}
    
    # 基本的な誤解パターン
    patterns['addition_subtraction_confusion'] = int('add' in text and 'subtract' in text)
    patterns['multiplication_division_confusion'] = int('multiply' in text and 'divide' in text)
    patterns['order_of_operations'] = int('order' in text or 'first' in text or 'then' in text)
    patterns['fraction_decimal_confusion'] = int('fraction' in text and 'decimal' in text)
    patterns['negative_number_issue'] = int('negative' in text or 'minus' in text)
    
    # 詳細なエラーパターン
    patterns['calculation_error'] = int('wrong' in text or 'incorrect' in text or 'mistake' in text)
    patterns['sign_error'] = int('positive' in text and 'negative' in text)
    patterns['carry_error'] = int('carry' in text or 'regroup' in text)
    patterns['place_value_error'] = int('place value' in text or 'place-value' in text)
    
    # 概念的エラー
    patterns['conceptual_confusion'] = int('confused' in text or 'mixed up' in text or 'thought' in text)
    patterns['procedural_error'] = int('step' in text and ('skip' in text or 'miss' in text or 'forgot' in text))
    patterns['interpretation_error'] = int('interpret' in text or 'understand' in text or 'misunderstood' in text)
    
    # 数学的誤解の具体的パターン
    patterns['fraction_whole_confusion'] = int('fraction' in text and 'whole' in text)
    patterns['percentage_decimal_error'] = int('percent' in text and 'decimal' in text)
    patterns['equals_means_calculate'] = int('equals' in text and ('calculate' in text or 'answer' in text))
    
    # 論理的エラー
    patterns['logic_error'] = int('because' in text and 'but' in text)
    patterns['assumption_error'] = int('assume' in text or 'thought' in text or 'believed' in text)
    patterns['reversed_operation'] = int('instead' in text or 'opposite' in text)
    
    return patterns

def extract_semantic_features(question, answer, explanation):
    """問題、解答、説明の意味的整合性を評価"""
    features = {}
    
    # 共通単語の割合
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    e_words = set(explanation.lower().split())
    
    features['q_a_word_overlap'] = len(q_words & a_words) / (len(q_words | a_words) + 1)
    features['q_e_word_overlap'] = len(q_words & e_words) / (len(q_words | e_words) + 1)
    features['a_e_word_overlap'] = len(a_words & e_words) / (len(a_words | e_words) + 1)
    features['all_overlap'] = len(q_words & a_words & e_words) / (len(q_words | a_words | e_words) + 1)
    
    # 数値の整合性
    q_numbers = set(re.findall(r'\b\d+\.?\d*\b', question))
    a_numbers = set(re.findall(r'\b\d+\.?\d*\b', answer))
    e_numbers = set(re.findall(r'\b\d+\.?\d*\b', explanation))
    
    features['q_e_number_overlap'] = len(q_numbers & e_numbers) / (len(q_numbers) + 1)
    features['answer_in_explanation'] = int(any(num in explanation for num in a_numbers))
    features['new_numbers_in_explanation'] = len(e_numbers - q_numbers - a_numbers)
    
    # キーワードの存在確認
    math_keywords = ['solve', 'calculate', 'find', 'determine', 'compute', 'equal', 'answer']
    features['has_action_word'] = int(any(kw in explanation.lower() for kw in math_keywords))
    
    # 説明の質
    features['explanation_completeness'] = len(e_words) / (len(q_words) + len(a_words) + 1)
    features['uses_question_terms'] = sum(1 for w in q_words if w in e_words) / (len(q_words) + 1)
    
    return features

def extract_question_type_features(question_text):
    """問題タイプの特徴"""
    features = {}
    
    # 問題のタイプ
    question_types = {
        'word_problem': ['john', 'mary', 'store', 'bought', 'has', 'gave', 'total', 'each'],
        'pure_math': ['solve', 'calculate', 'simplify', 'evaluate', 'find'],
        'geometry': ['triangle', 'circle', 'area', 'perimeter', 'angle', 'side', 'square'],
        'algebra': ['equation', 'variable', 'x', 'y', 'solve for', 'expression'],
        'fraction': ['fraction', 'numerator', 'denominator', 'mixed', 'improper', 'simplest'],
        'percentage': ['percent', '%', 'percentage', 'of', 'discount'],
        'measurement': ['meter', 'kilometer', 'mile', 'hour', 'minute', 'pound', 'kilogram'],
        'comparison': ['greater', 'less', 'more', 'fewer', 'compare', 'which'],
    }
    
    for qtype, keywords in question_types.items():
        features[f'is_{qtype}'] = int(any(kw in question_text.lower() for kw in keywords))
    
    # 問題の形式
    features['is_multiple_choice'] = int('which' in question_text.lower() or 'choose' in question_text.lower())
    features['is_fill_blank'] = int('___' in question_text or '...' in question_text or 'blank' in question_text)
    features['has_equation'] = int('=' in question_text)
    features['asks_for_explanation'] = int('explain' in question_text.lower() or 'why' in question_text.lower())
    
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
    df['answer_to_question_ratio'] = df['mc_answer_len'] / (df['question_len'] + 1)
    df['explanation_to_answer_ratio'] = df['explanation_len'] / (df['mc_answer_len'] + 1)
    
    # 数学的特徴を抽出
    for col in ['QuestionText', 'MC_Answer', 'StudentExplanation']:
        math_features = df[col].apply(extract_math_features).apply(pd.Series)
        prefix = 'q_' if col == 'QuestionText' else ('mc_' if col == 'MC_Answer' else 'exp_')
        math_features.columns = [f'{prefix}{c}' for c in math_features.columns]
        df = pd.concat([df, math_features], axis=1)
    
    # 統計的特徴を抽出
    for col in ['StudentExplanation', 'QuestionText']:
        stat_features = df[col].apply(extract_statistical_features).apply(pd.Series)
        prefix = 'exp_' if col == 'StudentExplanation' else 'q_'
        stat_features.columns = [f'{prefix}{c}' for c in stat_features.columns]
        df = pd.concat([df, stat_features], axis=1)
    
    # エラーパターン特徴
    error_features = df['StudentExplanation'].apply(extract_error_patterns).apply(pd.Series)
    error_features.columns = [f'error_{c}' for c in error_features.columns]
    df = pd.concat([df, error_features], axis=1)
    
    # 意味的特徴
    semantic_features = df.apply(lambda row: extract_semantic_features(
        row['QuestionText'], row['MC_Answer'], row['StudentExplanation']
    ), axis=1).apply(pd.Series)
    semantic_features.columns = [f'sem_{c}' for c in semantic_features.columns]
    df = pd.concat([df, semantic_features], axis=1)
    
    # 問題タイプ特徴
    qtype_features = df['QuestionText'].apply(extract_question_type_features).apply(pd.Series)
    qtype_features.columns = [f'qtype_{c}' for c in qtype_features.columns]
    df = pd.concat([df, qtype_features], axis=1)
    
    # 相互作用特徴
    df['total_numbers'] = df['q_number_count'] + df['mc_number_count'] + df['exp_number_count']
    df['total_operators'] = df['q_total_operators'] + df['mc_total_operators'] + df['exp_total_operators']
    df['math_complexity'] = df['total_numbers'] * df['total_operators']
    df['explanation_quality'] = df['exp_word_count'] * df['exp_unique_word_ratio']
    df['numeric_consistency'] = df['sem_q_e_number_overlap'] * df['sem_answer_in_explanation']
    
    # カテゴリ特徴
    if is_train:
        df['is_correct_answer'] = df['Category'].str.contains('True').astype(int)
        df['has_misconception'] = df['Category'].str.contains('Misconception').astype(int)
    
    return df

def post_process_predictions(predictions, test_df, train_misconception_counts):
    """予測の後処理"""
    processed_predictions = predictions.copy()
    
    # 問題タイプに基づく調整
    for idx, row in test_df.iterrows():
        # 簡単な問題（数値が少ない）の場合
        if row['q_number_count'] <= 2 and row['q_total_operators'] <= 1:
            # よくある基本的な誤解を優先
            # 実際の誤解IDに基づいて調整する必要がある
            pass
            
        # 分数問題の場合
        if row['qtype_is_fraction'] == 1:
            # 分数関連の誤解を優先
            pass
            
        # パーセント問題の場合
        if row['qtype_is_percentage'] == 1:
            # パーセント関連の誤解を優先
            pass
    
    return processed_predictions

if __name__ == '__main__':
    print('Starting optimized Claude model training...')
    
    # データ読み込み
    train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
    test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
    
    # Misconceptionの処理
    train['Misconception'] = train['Misconception'].fillna('NA').astype(str)
    train['target_cat'] = train.apply(lambda x: x['Category'] + ":" + x['Misconception'], axis=1)
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    
    # 誤解の頻度を計算（後処理用）
    train_misconception_counts = train['target_cat'].value_counts().to_dict()
    
    # 特徴量作成
    print("Creating features...")
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
    print("Cleaning text...")
    train['cleaned_text'] = train['combined_text'].apply(advanced_clean).apply(fast_lemmatize)
    test['cleaned_text'] = test['combined_text'].apply(advanced_clean).apply(fast_lemmatize)
    
    # TF-IDF特徴量（より高度な設定）
    print("Creating TF-IDF features...")
    
    # カスタムストップワード
    stop_words = list(stopwords.words('english'))
    stop_words.extend(['question', 'answer', 'explanation', 'student', 'the', 'a', 'an'])
    
    # TF-IDFベクトライザ（単一の高度な設定）
    tfidf = TfidfVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 5),  # 5-gramまで拡張
        max_df=0.9,
        min_df=2,
        max_features=12000,  # 特徴量を増やす
        token_pattern=r'\b\w+\b',
        sublinear_tf=True,  # TF-IDFのスケーリング
        use_idf=True,
        smooth_idf=True
    )
    
    # フィットと変換
    all_text = pd.concat([train['cleaned_text'], test['cleaned_text']])
    tfidf.fit(all_text)
    
    train_tfidf = tfidf.transform(train['cleaned_text'])
    test_tfidf = tfidf.transform(test['cleaned_text'])
    
    # 数値特徴量（拡張版）
    numeric_features = [col for col in train.columns if col not in 
                       ['QuestionId', 'QuestionText', 'MC_Answer', 'StudentExplanation', 
                        'Category', 'Misconception', 'target_cat', 'target_encoded', 
                        'combined_text', 'cleaned_text', 'is_correct_answer', 'has_misconception']]
    
    print(f"Number of numeric features: {len(numeric_features)}")
    
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
    
    # XGBoostモデル（単一の最適化された設定）
    print("Training XGBoost model...")
    nsplit = 5
    skf = StratifiedKFold(n_splits=nsplit, shuffle=True, random_state=42)
    
    # 最適化されたパラメータ
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
        'max_depth': 14,  # より深く
        'learning_rate': 0.025,  # 少し低め
        'subsample': 0.92,
        'colsample_bytree': 0.92,
        'colsample_bylevel': 0.65,
        'min_child_weight': 1.5,
        'gamma': 0.12,
        'alpha': 0.08,
        'lambda': 1.2,
        'random_state': 42,
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'max_bin': 512  # より細かいビン
    }
    
    oof_preds = np.zeros((len(train), n_classes))
    test_preds = np.zeros((len(test), n_classes))
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train_all, train['target_encoded'])):
        print(f"\nFold {fold + 1}")
        
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
            num_boost_round=2000,  # より多くのラウンド
            evals=[(dvalid, 'valid')],
            early_stopping_rounds=150,
            verbose_eval=100
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
    
    # テスト予測（後処理付き）
    test_preds = post_process_predictions(test_preds, test, train_misconception_counts)
    
    test_top3_indices = np.argsort(-test_preds, axis=1)[:, :3]
    test_predictions = []
    for indices in test_top3_indices:
        pred = [target_classes[i] for i in indices]
        test_predictions.append(' '.join(pred))
    
    # 提出ファイル作成
    submission = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")
    submission['Category:Misconception'] = test_predictions
    submission.to_csv("submission_claude_optimized.csv", index=False)
    print("\nSubmission file created: submission_claude_optimized.csv")
    
    # 特徴量の重要度を保存
    print("\nSaving feature importance...")
    importance = model.get_score(importance_type='gain')
    with open('feature_importance_claude.txt', 'w') as f:
        for feat, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:100]:
            f.write(f"{feat}: {score}\n")