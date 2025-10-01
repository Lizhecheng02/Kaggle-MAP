"""
Phi-4-mini-instruct モデル推論スクリプト - 提出用予測ファイルの生成
"""

import pandas as pd
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    AutoModelForCausalLM
)
from datasets import Dataset
import joblib
import torch
import torch.nn as nn
import gc
# PEFTのインポートをオプショナルにする
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available, will use base model only")

# カスタムモジュールのインポート
from config import *
from utils import prepare_correct_answers, format_input, tokenize_dataset, create_submission


# Phi-4-mini-instruct用のカスタム分類ヘッド
class Phi4ForSequenceClassification(nn.Module):
    """Phi-4-mini-instructモデルを分類タスク用にカスタマイズ"""
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.phi = base_model
        self.config = base_model.config
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def gradient_checkpointing_enable(self):
        """Gradient checkpointingを有効化"""
        if hasattr(self.phi, 'gradient_checkpointing_enable'):
            self.phi.gradient_checkpointing_enable()
            
    def enable_input_require_grads(self):
        """入力の勾配を要求"""
        if hasattr(self.phi, 'enable_input_require_grads'):
            self.phi.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.phi.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.phi(input_ids=input_ids, attention_mask=attention_mask)
        # 最後のトークンの隠れ状態を使用
        pooled_output = outputs.last_hidden_state[:, -1, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return type('Output', (), {'loss': loss, 'logits': logits})()


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

        # ベースモデルを読み込む（8bit量子化で読み込み）
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4"
        )
        
        # ベースモデルをCausalLMとして読み込む
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",  # 自動的に複数GPUに分散
            low_cpu_mem_usage=True,  # CPUメモリ使用量を削減
            attn_implementation="eager"  # Flash Attentionが対応していない場合の対処
        )
        
        # カスタム分類ヘッドを作成
        model = Phi4ForSequenceClassification(base_model, n_classes)
        
        # LoRAアダプターを適用
        model = PeftModel.from_pretrained(model, BEST_MODEL_PATH)
        
        # 推論モードに設定（メモリ効率化）
        model.eval()
        # 8bit量子化モデルは既にGPUに配置されているのでto('cuda')は不要

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
        model.base_model.config.pad_token_id = tokenizer.pad_token_id
        # 内部のモデルにも設定
        if hasattr(model.base_model, 'model'):
            model.base_model.model.config.pad_token_id = tokenizer.pad_token_id
    else:
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
