"""
Phi-4 階層モデル 推論スクリプト - 提出用予測ファイルの生成
"""

import pandas as pd
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding as HFDataCollatorWithPadding,
)
from datasets import Dataset
import joblib
import torch
import gc
from models import HierPhi4ForSequenceClassification
from utils import prepare_correct_answers, format_input, tokenize_dataset, create_submission
from config import *

# PEFTのインポートをオプショナルにする
try:
    from peft import PeftModel
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

    print("Loading label encoders and hierarchical meta...")
    le = joblib.load(LABEL_ENCODER_PATH)  # joint
    hier_meta = joblib.load(HIER_META_PATH)
    n_classes = len(le.classes_)

    print("Loading trained model and tokenizer...")

    if not PEFT_AVAILABLE:
        raise ImportError("PEFT is required to load the fine-tuned model. Please install peft: pip install peft")

    print(f"Loading base model from: {MODEL_NAME}")
    base = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = 100257
    base.config.pad_token_id = tokenizer.pad_token_id

    model_core = HierPhi4ForSequenceClassification(
        backbone=base,
        hidden_size=base.config.hidden_size,
        n_joint=n_classes,
        n_cat=len(hier_meta['le_cat'].classes_),
        n_mc=len(hier_meta['le_mc'].classes_),
        joint_to_cat=hier_meta['joint_to_cat'],
        joint_to_mc=hier_meta['joint_to_mc'],
        cat_is_misconc=hier_meta['cat_is_misconc'],
        mc_na_index=int(hier_meta['mc_na_index']),
        lambda_cat=0.0,
        lambda_mc=0.0,
        lambda_constraint=0.0,
    )

    print(f"Loading LoRA adapter from: {BEST_MODEL_PATH}")
    model = PeftModel.from_pretrained(model_core, BEST_MODEL_PATH)
    model.eval()

    # PeftModel 内のバックボーンにも設定
    try:
        model.base_model.backbone.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass

    print("Loading test data...")
    # テストデータの読み込み
    test = pd.read_csv(TEST_DATA_PATH)

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

    # パディングのためのデータコラレータの設定（HF版でOK）
    data_collator = HFDataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')

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
