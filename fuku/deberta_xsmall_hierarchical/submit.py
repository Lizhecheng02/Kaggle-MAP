"""
階層的分類アプローチの推論スクリプト
"""

import pandas as pd
import numpy as np
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import joblib
import torch
import os

# wandbを無効化
os.environ["WANDB_DISABLED"] = "true"

# カスタムモジュールのインポート
from config import *
from utils import *


def predict_category(test_data, tokenizer, model_path, encoder_path):
    """Categoryの予測"""
    print("Predicting categories...")

    # エンコーダのロード
    category_encoder = joblib.load(encoder_path)

    # モデルのロード
    model = DebertaV2ForSequenceClassification.from_pretrained(model_path)

    # 入力テキストのフォーマット
    test_data['text'] = test_data.apply(
        lambda x: format_input_hierarchical(x, task='category'),
        axis=1
    )

    # データセットの作成
    test_dataset = Dataset.from_pandas(test_data[['text']])
    test_dataset = tokenize_dataset(test_dataset, tokenizer, MAX_LEN)

    # トレーナーの初期化（推論用）
    # wandbを無効化するための引数を設定
    training_args = TrainingArguments(
        output_dir="./tmp_trainer",
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        report_to="none",
        logging_steps=500000,  # 大きな値を設定してロギングを事実上無効化
        disable_tqdm=False
    )
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args
    )

    # 予測
    predictions = trainer.predict(test_dataset)

    return predictions.predictions, category_encoder


def predict_misconception(test_data, category, tokenizer, model_path, encoder_path):
    """特定CategoryのMisconceptionの予測"""
    print(f"Predicting misconceptions for {category}...")

    # エンコーダのロード
    misconception_encoder = joblib.load(encoder_path)

    # モデルのロード
    model = DebertaV2ForSequenceClassification.from_pretrained(model_path)

    # 該当するCategoryのデータのみ抽出
    category_mask = test_data['predicted_category'] == category
    subset_data = test_data[category_mask].copy()

    if len(subset_data) == 0:
        return None, misconception_encoder

    # 入力テキストのフォーマット
    subset_data['text'] = subset_data.apply(
        lambda x: format_input_hierarchical(x, task='misconception'),
        axis=1
    )

    # データセットの作成
    test_dataset = Dataset.from_pandas(subset_data[['text']])
    test_dataset = tokenize_dataset(test_dataset, tokenizer, MAX_LEN)

    # トレーナーの初期化（推論用）
    # wandbを無効化するための引数を設定
    training_args = TrainingArguments(
        output_dir="./tmp_trainer",
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        report_to="none",
        logging_steps=500000,  # 大きな値を設定してロギングを事実上無効化
        disable_tqdm=False
    )
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args
    )

    # 予測
    predictions = trainer.predict(test_dataset)

    # 全データ用の予測配列を作成
    full_predictions = np.zeros((len(test_data), len(misconception_encoder.classes_)))
    full_predictions[category_mask] = predictions.predictions

    return full_predictions, misconception_encoder


def main():
    """メイン関数"""

    # データの読み込み
    print("Loading test data...")
    test = pd.read_csv(TEST_DATA_PATH)

    # 特徴量エンジニアリング（推論時は仮の正解ラベルを使用）
    print("Performing feature engineering...")
    # テストデータでは正解がわからないので、全て不正解と仮定
    test['is_correct'] = 0

    # トークナイザーの初期化
    print("Initializing tokenizer...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

    # 1. Category予測
    category_predictions, category_encoder = predict_category(
        test, tokenizer, CATEGORY_MODEL_PATH, CATEGORY_ENCODER_PATH
    )

    # Category予測結果を追加
    category_probs = torch.nn.functional.softmax(torch.tensor(category_predictions), dim=1).numpy()
    category_preds = np.argmax(category_probs, axis=1)
    test['predicted_category'] = category_encoder.inverse_transform(category_preds)

    # 2. 各CategoryのMisconception予測
    misconception_predictions = {}
    misconception_encoders = {}

    # True_Misconception
    if 'True_Misconception' in test['predicted_category'].values:
        preds, encoder = predict_misconception(
            test, 'True_Misconception', tokenizer,
            TRUE_MISCONCEPTION_MODEL_PATH, TRUE_MISCONCEPTION_ENCODER_PATH
        )
        if preds is not None:
            misconception_predictions['True_Misconception'] = preds
            misconception_encoders['True_Misconception'] = encoder

    # False_Misconception
    if 'False_Misconception' in test['predicted_category'].values:
        preds, encoder = predict_misconception(
            test, 'False_Misconception', tokenizer,
            FALSE_MISCONCEPTION_MODEL_PATH, FALSE_MISCONCEPTION_ENCODER_PATH
        )
        if preds is not None:
            misconception_predictions['False_Misconception'] = preds
            misconception_encoders['False_Misconception'] = encoder

    # 3. 提出ファイルの作成
    print("Creating submission file...")
    submission = create_hierarchical_submission(
        category_predictions,
        misconception_predictions,
        test,
        category_encoder,
        misconception_encoders
    )

    # 提出ファイルの保存
    submission_path = f"submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")

    # サンプル表示
    print("\nSample predictions:")
    print(submission.head(10))


if __name__ == "__main__":
    main()
