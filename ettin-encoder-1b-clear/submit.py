"""
Deberta モデル推論スクリプト - 提出用予測ファイルの生成
"""

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import joblib

# カスタムモジュールのインポート
from config import *
from utils import prepare_correct_answers, format_input, tokenize_dataset, create_submission, preprocess_data


def main():
    """メイン推論関数"""

    print("Loading trained model and tokenizer...")
    # モデルとトークナイザーの読み込み
    model = AutoModelForSequenceClassification.from_pretrained(
        BEST_MODEL_PATH,
        reference_compile=REFERENCE_COMPILE,
    )
    tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_PATH)

    print("Loading label encoder...")
    # ラベルエンコーダーの読み込み
    le = joblib.load(LABEL_ENCODER_PATH)

    print("Loading test data...")
    # テストデータの読み込み
    test = pd.read_csv(TEST_DATA_PATH)

    print("Loading training data for correct answers...")
    # 正解答案データの準備（訓練データから取得）
    train = pd.read_csv(TRAIN_DATA_PATH)
    
    # preprocess_data関数を適用
    print("Applying data preprocessing to train data...")
    train = preprocess_data(train)
    
    correct = prepare_correct_answers(train)

    print("Preprocessing test data...")
    # テストデータの前処理
    # preprocess_data関数を適用
    test = preprocess_data(test)
    
    test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    test.is_correct = test.is_correct.fillna(0)
    
    # FixedStudentExplanationを使用するようにformat_inputを適用
    test_for_format = test.copy()
    test_for_format['StudentExplanation'] = test['FixedStudentExplanation']
    test['text'] = test_for_format.apply(format_input, axis=1)

    print("Tokenizing test data...")
    # テストデータのトークナイズ
    ds_test = Dataset.from_pandas(test[['text']])
    ds_test = tokenize_dataset(ds_test, tokenizer, MAX_LEN)

    print("Running inference...")
    # 推論の実行
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(
            output_dir="./tmp",  # 一時ディレクトリ（必須パラメータ）
            report_to="none",    # wandbを無効化
            per_device_eval_batch_size=EVAL_BATCH_SIZE,  # バッチサイズの設定
            bf16=BF16,
            fp16=FP16,
        )
    )
    predictions = trainer.predict(ds_test)

    print("Creating submission file...")
    # 提出用ファイルの作成
    submission = create_submission(predictions, test, le)

    # ファイルの保存
    submission.to_csv(SUBMISSION_OUTPUT_PATH, index=False)
    print(f"Submission file saved to: {SUBMISSION_OUTPUT_PATH}")
    print("\nSubmission preview:")
    print(submission.head())
    print(f"\nSubmission shape: {submission.shape}")


if __name__ == "__main__":
    main()
