"""
Qwen-3-0.6B モデル推論スクリプト - 提出用予測ファイルの生成
"""

import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import joblib
import torch
import os
import gc
# メモリ最適化のための環境変数設定
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# PEFTのインポートをオプショナルにする
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available, will use base model only")

from config import *  # 設定ファイルから定数をインポート
from utils import prepare_correct_answers, format_input, tokenize_dataset, DataCollatorWithPadding, create_submission



# カスタムクラスの定義（train.pyと同じもの）
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

class Qwen2VLForSequenceClassification(nn.Module):
    """Qwen2.5-VLモデルを分類タスク用にカスタマイズ"""
    def __init__(self, model_name, num_labels):
        super().__init__()
        try:
            from transformers import Qwen2VLModel
            self.qwen = Qwen2VLModel.from_pretrained(model_name, trust_remote_code=True)
        except:
            # Qwen2VLModelが利用できない場合は、AutoModelを使用
            from transformers import AutoModel
            self.qwen = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # configを保存
        self.config = self.qwen.config
        # hidden_sizeを動的に取得
        if hasattr(self.qwen.config, 'hidden_size'):
            hidden_size = self.qwen.config.hidden_size
        elif hasattr(self.qwen.config, 'd_model'):
            hidden_size = self.qwen.config.d_model
        else:
            # デフォルト値（Qwen2.5-VL-3Bの場合）
            hidden_size = 1536

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, **kwargs):
        # input_idsまたはinputs_embedsのいずれかを使用
        if input_ids is not None:
            outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        elif inputs_embeds is not None:
            outputs = self.qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # 最後のトークンの隠れ状態を使用
        pooled_output = outputs.last_hidden_state[:, -1, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )


def main():
    """メイン推論関数"""

    # CUDAメモリの最適化
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        
        # GPUメモリの断片化を防ぐ
        torch.cuda.set_per_process_memory_fraction(0.95, 0)
        torch.cuda.set_per_process_memory_fraction(0.95, 1)

    print("Loading label encoder...")
    # ラベルエンコーダーの読み込み
    le = joblib.load(LABEL_ENCODER_PATH)
    n_classes = len(le.classes_)

    print("Loading trained model and tokenizer...")

    if PEFT_AVAILABLE:
        # LoRAアダプターを使用する場合
        print(f"Loading fine-tuned LoRA model from: {BEST_MODEL_PATH}")
        print(f"Loading base model from: {MODEL_NAME}")

        try:
            # まず通常のAutoModelForSequenceClassificationを試す
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=n_classes,
                trust_remote_code=True,
                device_map="balanced",  # より効率的な分散
                torch_dtype=torch.float16,  # メモリ節約のためfp16を使用
                low_cpu_mem_usage=True,  # CPU使用量を削減
                max_memory={0: "13GB", 1: "13GB", "cpu": "20GB"},  # より保守的な設定
                load_in_8bit=True,  # 8bit量子化でメモリ使用量を半減
                offload_folder="offload",  # 必要に応じてCPUにオフロード
                offload_state_dict=True,  # state_dictもオフロード
            )
        except Exception as e:
            print(f"8bit loading failed: {e}")
            print("Trying without 8bit quantization...")
            try:
                # 8bit量子化なしで再試行
                model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME,
                    num_labels=n_classes,
                    trust_remote_code=True,
                    device_map="balanced",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    max_memory={0: "13GB", 1: "13GB", "cpu": "20GB"},
                    offload_folder="offload",
                    offload_state_dict=True,
                )
            except Exception as e2:
                # それでも失敗した場合はカスタムクラスを使用
                print(f"Standard model loading also failed: {e2}")
                print("Using custom classification head for Qwen2.5-VL...")
                # カスタムモデルもfp16で初期化
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    model = Qwen2VLForSequenceClassification(MODEL_NAME, n_classes)
                model = model.half()  # fp16に変換

        # LoRAアダプターを適用
        model = PeftModel.from_pretrained(model, BEST_MODEL_PATH)

        # トークナイザーはベースモデルから読み込む
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print("Successfully loaded LoRA fine-tuned model")
    else:
        # PEFTが利用できない場合はエラー
        raise ImportError("PEFT is required to load the fine-tuned model. Please install peft: pip install peft")

    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # モデルの設定を更新（PeftModelのbase_modelにアクセス）
    if hasattr(model, 'base_model'):
        # PeftModelの場合
        if hasattr(model.base_model, 'model'):
            # model.base_model.modelが実際のモデル
            if hasattr(model.base_model.model, 'config'):
                model.base_model.model.config.pad_token_id = tokenizer.pad_token_id
        elif hasattr(model.base_model, 'config'):
            # model.base_modelが実際のモデル
            model.base_model.config.pad_token_id = tokenizer.pad_token_id
    elif hasattr(model, 'config'):
        # 通常のモデル
        model.config.pad_token_id = tokenizer.pad_token_id

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

    # パディングのためのデータコラレータの設定
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Running inference...")
    # 推論実行前にメモリをクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # 推論の実行
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,  # tokenizer の代替
        data_collator=data_collator,  # バッチ時に自動でパディングを適用
        args=TrainingArguments(
            output_dir="./tmp",  # 一時ディレクトリ（必須パラメータ）
            report_to="none",    # wandbを無効化
            per_device_eval_batch_size=1,  # 最小バッチサイズ
            fp16=True,  # メモリ効率のため追加
            dataloader_num_workers=0,  # ワーカーを使わない
            dataloader_pin_memory=False,  # pin_memoryを無効化してメモリ節約
            gradient_checkpointing=False,  # 推論時は不要
            eval_accumulation_steps=10,  # 予測結果を定期的にCPUに移動
        )
    )
    # バッチ処理でメモリ効率化
    try:
        predictions = trainer.predict(ds_test)
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory! Reducing batch size and retrying...")
        torch.cuda.empty_cache()
        gc.collect()

        # バッチサイズをさらに削減して再試行
        trainer.args.per_device_eval_batch_size = 1
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
