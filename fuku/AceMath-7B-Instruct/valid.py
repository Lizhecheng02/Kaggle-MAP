"""
全トレーニングデータに対して推論を実行し、確率を出力するスクリプト
対象モデル: AceMath-7B-Instruct
出力: numpy配列 (n, C) - n: トレーニングデータの行数, C: クラス数
"""

import os
import sys
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding, BitsAndBytesConfig
from datasets import Dataset
from peft import PeftModel
import joblib
from tqdm import tqdm

# 設定ファイルをインポート
from config import *
from utils import format_input, prepare_correct_answers, tokenize_dataset

# tokenize_dataset(utils側) を用いるためローカル関数は削除

def load_model_and_tokenizer():
    """モデルとトークナイザーをロード"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # パディングトークンの設定（モデル依存の特殊トークン固定を避ける）
    if tokenizer.pad_token is None:
        if PAD_TOKEN_STR is not None and PAD_TOKEN_ID is not None:
            tokenizer.add_special_tokens({"pad_token": PAD_TOKEN_STR})
            tokenizer.pad_token_id = PAD_TOKEN_ID
        elif USE_EOS_AS_PAD and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    
    print("Loading label encoder...")
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    n_classes = len(label_encoder.classes_)
    
    print(f"Loading model with {n_classes} classes...")
    
    use_cuda = torch.cuda.is_available()
    # 8bit量子化設定（GPUがある場合のみ）
    quantization_config = None
    if use_cuda:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4"
        )
    # ベースモデルのロード
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_classes,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map=("auto" if use_cuda else None),
        torch_dtype=(torch.float16 if use_cuda else torch.float32),
        low_cpu_mem_usage=True  # CPUメモリ使用量を削減
    )
    
    # LoRAアダプターを適用
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, BEST_MODEL_PATH)
    
    # モデルの設定を更新してパディングトークンを認識させる
    if hasattr(model, 'base_model'):
        model.base_model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model.base_model, 'model'):
            model.base_model.model.config.pad_token_id = tokenizer.pad_token_id
    else:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # モデルを評価モードに設定
    model.eval()
    
    # 8bit量子化モデルは既にGPUに配置されているのでto('cuda')は不要
    if torch.cuda.is_available():
        print("Using GPU for inference")
    else:
        print("Using CPU for inference")
    
    return model, tokenizer, label_encoder

def prepare_data():
    """トレーニングデータを準備"""
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    
    # train.pyと同じ前処理を適用
    print("Performing feature engineering...")
    correct = prepare_correct_answers(train_df)
    train_df = train_df.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    train_df.is_correct = train_df.is_correct.fillna(0)
    
    # プロンプトを作成
    train_df['text'] = train_df.apply(format_input, axis=1)
    
    # データセットに変換
    dataset = Dataset.from_pandas(train_df[['text']])
    
    return train_df, dataset

def inference_with_batches(model, tokenizer, dataset, batch_size=16):
    """バッチ処理で推論を実行"""
    # データコレーターの設定
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')

    # トークナイズ（チャットテンプレート適用を含む）
    print("Tokenizing dataset (with chat template)...")
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, MAX_LEN)
    
    # Trainerを使用した推論（メモリ効率的）
    use_fp16 = torch.cuda.is_available()
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
        args=TrainingArguments(
            output_dir="./tmp_valid",
            per_device_eval_batch_size=batch_size,
            fp16=use_fp16,
            dataloader_num_workers=2,
            remove_unused_columns=True,
        )
    )
    
    print(f"Running inference with batch size {batch_size}...")
    predictions = trainer.predict(tokenized_dataset)
    
    # ロジットから確率に変換
    logits = predictions.predictions
    probabilities = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()
    
    return probabilities

def main():
    """メイン処理"""
    # メモリキャッシュをクリア
    torch.cuda.empty_cache()
    gc.collect()
    
    # CUDAメモリ管理の最適化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # TF32を有効化（推論速度向上）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # モデルとトークナイザーをロード
    model, tokenizer, label_encoder = load_model_and_tokenizer()
    
    # データを準備
    train_df, dataset = prepare_data()
    
    # バッチサイズを設定（メモリに応じて調整）
    batch_size = EVAL_BATCH_SIZE
    if torch.cuda.is_available():
        # GPUメモリに余裕がある場合は増やす
        batch_size = 16
    
    # 推論実行
    try:
        with torch.no_grad():
            probabilities = inference_with_batches(model, tokenizer, dataset, batch_size)
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory. Reducing batch size...")
        torch.cuda.empty_cache()
        batch_size = max(1, batch_size // 2)
        with torch.no_grad():
            probabilities = inference_with_batches(model, tokenizer, dataset, batch_size)
    
    # 結果の確認
    print(f"\nInference completed!")
    print(f"Shape of probabilities: {probabilities.shape}")
    print(f"Expected shape: ({len(train_df)}, {len(label_encoder.classes_)})")
    
    # numpy配列として保存
    output_path = os.path.join(OUTPUT_DIR, "train_probabilities.npy")
    np.save(output_path, probabilities)
    print(f"\nProbabilities saved to: {output_path}")
    
    # 確認用：最初の5行の上位3クラスを表示
    print("\nSample predictions (top 3 classes for first 5 rows):")
    for i in range(min(5, len(probabilities))):
        top_indices = np.argsort(probabilities[i])[::-1][:3]
        top_probs = probabilities[i][top_indices]
        top_labels = label_encoder.inverse_transform(top_indices)
        print(f"Row {i}: {list(zip(top_labels, top_probs))}")
    
    # メモリ使用量の情報
    if torch.cuda.is_available():
        print(f"\nGPU memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    return probabilities

if __name__ == "__main__":
    probabilities = main()
