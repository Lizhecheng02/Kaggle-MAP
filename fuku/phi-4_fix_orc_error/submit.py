"""
Phi-4 モデル推論スクリプト - 提出用予測ファイルの生成
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
import gc
# PEFTのインポートをオプショナルにする
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available, will use base model only")


# Kaggle環境ではカスタムクラスは不要


def main():
    """メイン推論関数"""

    # メモリキャッシュをクリア
    torch.cuda.empty_cache()
    gc.collect()

    # CUDAメモリ管理の最適化
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 2つのGPUを使用可能にする
    if torch.cuda.device_count() > 1:
        print(f"Found {torch.cuda.device_count()} GPUs")

    print("Loading label encoder...")
    # ラベルエンコーダーの読み込み
    le = joblib.load(LABEL_ENCODER_PATH)
    n_classes = len(le.classes_)

    print("Loading trained model and tokenizer...")

    if PEFT_AVAILABLE:
        # LoRAアダプターを使用する場合
        print(f"Loading fine-tuned LoRA model from: {BEST_MODEL_PATH}")
        print(f"Loading base model from: {MODEL_NAME}")

        # ベースモデルを読み込む（量子化なしでフルプレシジョン）
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=n_classes,
            trust_remote_code=True,
            device_map="auto",  # 自動的に複数GPUに分散
            torch_dtype=torch.float16,  # float16を使用（メモリ効率とパフォーマンスのバランス）
            low_cpu_mem_usage=True  # CPUメモリ使用量を削減
        )

        # LoRAアダプターを適用
        model = PeftModel.from_pretrained(model, BEST_MODEL_PATH)

        # 推論モードに設定（メモリ効率化）
        model.eval()
        # モデルは既にdevice_mapでGPUに配置されているのでto('cuda')は不要

        # トークナイザーはベースモデルから読み込む
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print("Successfully loaded LoRA fine-tuned model")
    else:
        # PEFTが利用できない場合はエラー
        raise ImportError("PEFT is required to load the fine-tuned model. Please install peft: pip install peft")

    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = 100257

    # モデルの設定を更新（PeftModelのbase_modelにアクセス）
    if hasattr(model, 'base_model'):
        model.base_model.config.pad_token_id = tokenizer.pad_token_id
        # 内部のモデルにも設定
        if hasattr(model.base_model, 'model'):
            model.base_model.model.config.pad_token_id = tokenizer.pad_token_id
    else:
        model.config.pad_token_id = tokenizer.pad_token_id

    print("Loading test data...")
    # テストデータの読み込み
    test = pd.read_csv(TEST_DATA_PATH)

    print("Creating OCR correction dictionary...")
    # OCRエラー修正用辞書の作成
    ocr_correction_dict = create_ocr_correction_dict(OCR_CORRECTION_CSV_PATH)
    print(f"Loaded {len(ocr_correction_dict)} OCR correction patterns")

    print("Applying OCR error corrections to test data...")
    # StudentExplanation列のOCRエラー修正
    if 'StudentExplanation' in test.columns:
        test['StudentExplanation'] = test['StudentExplanation'].apply(
            lambda x: fix_ocr_errors(x, ocr_correction_dict) if pd.notna(x) else x
        )

    # QuestionText列のOCRエラー修正（存在する場合）
    if 'QuestionText' in test.columns:
        test['QuestionText'] = test['QuestionText'].apply(
            lambda x: fix_ocr_errors(x, ocr_correction_dict) if pd.notna(x) else x
        )

    print("Loading training data for correct answers...")
    # 正解答案データの準備（訓練データから取得）
    train = pd.read_csv(TRAIN_DATA_PATH)
    train.Misconception = train.Misconception.fillna('NA')
    correct = prepare_correct_answers(train)

    print("Preprocessing test data...")
    # テストデータの前処理
    test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    test.is_correct = test.is_correct.fillna(0)
    test['text'] = test.apply(format_input, axis=1)

    print("Tokenizing test data...")
    # テストデータのトークナイズ
    ds_test = Dataset.from_pandas(test[['text']])
    ds_test = tokenize_dataset(ds_test, tokenizer, MAX_LEN)

    # パディングのためのデータコラレータの設定
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Running inference...")

    # TF32を有効化（推論速度向上）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 推論の実行
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,  # tokenizer の代替
        data_collator=data_collator,  # バッチ時に自動でパディングを適用
        args=TrainingArguments(
            output_dir="./tmp",  # 一時ディレクトリ（必須パラメータ）
            report_to="none",    # wandbを無効化
            per_device_eval_batch_size=EVAL_BATCH_SIZE,  # 設定ファイルから取得
            fp16=True,  # float16を使用
            dataloader_pin_memory=True,  # データローダーの高速化
            dataloader_num_workers=2,  # データ読み込みの並列化
        )
    )
    # no_gradコンテキストで推論を実行（メモリ効率化）
    with torch.no_grad():
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
