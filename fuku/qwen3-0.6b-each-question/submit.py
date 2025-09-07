"""
Qwen-3-0.6B モデル推論スクリプト - 提出用予測ファイルの生成
"""

import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import joblib
import torch
# PEFTのインポートをオプショナルにする
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available, will use base model only")

# カスタムモジュールのインポート
from config import *
from utils import (
    prepare_correct_answers, tokenize_dataset, create_submission,
    filter_data_by_question_id, create_combined_submission
)
from prompts import prompt_registry


# Kaggle環境ではカスタムクラスは不要


def load_question_model(question_id):
    """特定のQuestionId用のモデルとラベルエンコーダーを読み込む"""
    
    question_model_path = get_question_model_path(question_id)
    question_label_encoder_path = get_question_label_encoder_path(question_id)
    
    if not os.path.exists(question_label_encoder_path):
        print(f"Warning: Label encoder for Question {question_id} not found at {question_label_encoder_path}")
        return None, None, None
    
    if not os.path.exists(question_model_path):
        print(f"Warning: Model for Question {question_id} not found at {question_model_path}")
        return None, None, None
    
    print(f"Loading model for Question {question_id}...")
    
    # ラベルエンコーダーの読み込み
    le = joblib.load(question_label_encoder_path)
    n_classes = len(le.classes_)
    
    if PEFT_AVAILABLE:
        # ベースモデルを読み込む
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=n_classes,
            trust_remote_code=True
        )
        
        # LoRAアダプターを適用
        model = PeftModel.from_pretrained(model, question_model_path)
        
        # トークナイザーを読み込む
        tokenizer = AutoTokenizer.from_pretrained(question_model_path, trust_remote_code=True)
        
        # パディングトークンの設定
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # モデルの設定を更新
        if hasattr(model, 'base_model'):
            model.base_model.config.pad_token_id = tokenizer.pad_token_id
            if hasattr(model.base_model, 'model'):
                model.base_model.model.config.pad_token_id = tokenizer.pad_token_id
        else:
            model.config.pad_token_id = tokenizer.pad_token_id
        
        print(f"Successfully loaded model for Question {question_id} ({n_classes} classes)")
        return model, tokenizer, le
    else:
        raise ImportError("PEFT is required to load the fine-tuned model. Please install peft: pip install peft")


def predict_question(question_id, question_test_data, model, tokenizer, le):
    """特定のQuestionIdのテストデータに対して推論を実行"""
    
    if len(question_test_data) == 0:
        return None
    
    print(f"Running inference for Question {question_id} ({len(question_test_data)} samples)...")
    
    # プロンプト関数を設定から取得して使用
    prompt_function = prompt_registry[PROMPT_VERSION]
    question_test_data = question_test_data.copy()
    question_test_data['text'] = question_test_data.apply(lambda row: prompt_function(tokenizer, row), axis=1)
    
    # テストデータのトークナイズ
    ds_test = Dataset.from_pandas(question_test_data[['text']])
    ds_test = tokenize_dataset(ds_test, tokenizer, MAX_LEN)
    
    # パディングのためのデータコラレータの設定
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 推論の実行
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
        args=TrainingArguments(
            output_dir="./tmp",
            report_to="none",
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            fp16=True,
        )
    )
    predictions = trainer.predict(ds_test)
    
    return predictions


def main():
    """メイン推論関数 - QuestionIdごとの推論"""

    print("Loading test data...")
    test = pd.read_csv(TEST_DATA_PATH)
    
    print("Loading training data for correct answers...")
    train = pd.read_csv(TRAIN_DATA_PATH)
    train.Misconception = train.Misconception.fillna('NA')
    correct = prepare_correct_answers(train)
    
    print("Preprocessing test data...")
    test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    test.is_correct = test.is_correct.fillna(0)
    
    print(f"Test data shape: {test.shape}")
    print(f"Unique QuestionIds in test: {test['QuestionId'].nunique()}")
    
    # QuestionIdごとの予測結果を格納
    question_predictions_dict = {}
    successful_questions = []
    failed_questions = []
    
    print(f"\nStarting inference for {len(QUESTION_IDS)} questions...")
    
    for i, question_id in enumerate(QUESTION_IDS):
        print(f"\n{'='*60}")
        print(f"Processing Question {i+1}/{len(QUESTION_IDS)}: {question_id}")
        print(f"{'='*60}")
        
        try:
            # QuestionIdごとのテストデータをフィルタリング
            question_test_data = filter_data_by_question_id(test, question_id)
            
            if len(question_test_data) == 0:
                print(f"Warning: No test data found for Question {question_id}. Skipping...")
                failed_questions.append(question_id)
                continue
            
            # モデルとラベルエンコーダーを読み込む
            model, tokenizer, le = load_question_model(question_id)
            
            if model is None:
                print(f"Failed to load model for Question {question_id}. Skipping...")
                failed_questions.append(question_id)
                continue
            
            # 推論を実行
            predictions = predict_question(question_id, question_test_data, model, tokenizer, le)
            
            if predictions is not None:
                question_predictions_dict[question_id] = (predictions, question_test_data, le)
                successful_questions.append(question_id)
                print(f"✅ Question {question_id} inference completed successfully")
            else:
                failed_questions.append(question_id)
                print(f"⚠️  Question {question_id} inference failed")
            
            # メモリ解放
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            failed_questions.append(question_id)
            print(f"❌ Question {question_id} failed with error: {str(e)}")
            continue
    
    # 結果のサマリーを表示
    print(f"\n{'='*60}")
    print("INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions: {len(QUESTION_IDS)}")
    print(f"Successfully predicted: {len(successful_questions)}")
    print(f"Failed/Skipped: {len(failed_questions)}")
    
    if successful_questions:
        print(f"\nSuccessful questions: {successful_questions}")
    
    if failed_questions:
        print(f"\nFailed/Skipped questions: {failed_questions}")
    
    # 予測結果を統合して提出用ファイルを作成
    if question_predictions_dict:
        print("\nCreating combined submission file...")
        submission = create_combined_submission(question_predictions_dict, test)
        
        # ファイルの保存
        submission.to_csv(SUBMISSION_OUTPUT_PATH, index=False)
        print(f"Submission file saved to: {SUBMISSION_OUTPUT_PATH}")
        print("\nSubmission preview:")
        print(submission.head())
        print(f"\nSubmission shape: {submission.shape}")
    else:
        print("No successful predictions to create submission file!")


if __name__ == "__main__":
    main()
