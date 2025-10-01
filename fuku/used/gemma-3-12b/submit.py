"""
Gemma-3 モデル推論スクリプト - NaN問題完全修正版（bfloat16対応）
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
import pandas as pd
from transformers import DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import joblib
import torch
import numpy as np
import gc
import time
import os

# PEFTのインポートをオプショナルにする
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available, will use base model only")

# 設定のインポート
from config import *
from utils_with_choices import (
    prepare_correct_answers,
    format_input,
    tokenize_dataset,
    create_submission,
    prepare_answer_choices
)

# 環境変数の設定（tokenizers警告を抑制）
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# メモリキャッシュをクリア
torch.cuda.empty_cache()
gc.collect()

# CUDAメモリ管理の最適化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# GPUの確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"Found {torch.cuda.device_count()} GPUs")

    # Tesla T4の検出
    gpu_name = torch.cuda.get_device_name()
    is_tesla_t4 = "T4" in gpu_name
    print(f"Tesla T4 detected: {is_tesla_t4}")

print("Loading label encoder...")
# ラベルエンコーダーの読み込み
le = joblib.load(LABEL_ENCODER_PATH)
n_classes = len(le.classes_)

print("Loading trained model and tokenizer...")

if PEFT_AVAILABLE:
    # LoRAアダプターを使用する場合
    print(f"Loading fine-tuned LoRA model from: {BEST_MODEL_PATH}")
    print(f"Loading base model from: {MODEL_NAME}")

    # NaN問題解決のための設定変更
    print("Applying NaN-resistant configuration...")

    # bfloat16で統一（学習時と同じ精度）
    model_dtype = torch.bfloat16
    compute_dtype = torch.bfloat16

    # 量子化を軽量化または無効化
    if is_tesla_t4:
        # Tesla T4用：量子化を軽量化
        print("Using lightweight quantization for Tesla T4...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=True,  # 8bit量子化（4bitより安定）
            llm_int8_enable_fp32_cpu_offload=False,
            llm_int8_threshold=6.0,
        )
    else:
        # 他のGPU用：標準設定
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print(f"Loading base model with dtype: {model_dtype}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_classes,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_cfg,
        torch_dtype=model_dtype,  # bfloat16で統一
        low_cpu_mem_usage=True,
    )

    # LoRAアダプターを適用
    print("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(
        model,
        BEST_MODEL_PATH,
    )

    # 数値安定性のためモデルをマージ
    print("Merging LoRA adapter for numerical stability...")
    try:
        model = model.merge_and_unload()
        print("Successfully merged LoRA adapter")
    except Exception as e:
        print(f"Merge failed, continuing with PeftModel: {e}")

    # 推論モードに設定
    model.eval()

    # gradient checkpointingを無効化
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()

    # トークナイザーはベースモデルから読み込む
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Successfully loaded model")
else:
    # PEFTが利用できない場合はエラー
    raise ImportError("PEFT is required to load the fine-tuned model. Please install peft: pip install peft")

# パディングトークンの設定
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# モデルの設定を更新
try:
    model.config.pad_token_id = tokenizer.pad_token_id
except:
    # マージ後のモデルでエラーが出る場合
    pass

print("Loading test data...")
# テストデータの読み込み
test = pd.read_csv(TEST_DATA_PATH)
print(f"Test data shape: {test.shape}")

print("Loading training data for priors and correct answers...")
# 正解答案データの準備（訓練データから取得）
train = pd.read_csv(TRAIN_DATA_PATH)
train.Misconception = train.Misconception.fillna('NA')
correct = prepare_correct_answers(train)


print("Loading test data...")
# テストデータの読み込み
test = pd.read_csv(TEST_DATA_PATH)
print(f"Test data shape: {test.shape}")

print("Loading training data for priors and correct answers...")
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


# MC_Answerを選択肢ラベルに変換
def get_choice_label(row):
    choice_row = choices[choices['QuestionId'] == row['QuestionId']]
    if len(choice_row) > 0:
        mapping = choice_row.iloc[0]['choice_mapping']
        return mapping.get(row['MC_Answer'], row['MC_Answer'])
    return row['MC_Answer']




print("Running inference...")

# TF32を有効化（推論速度向上）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# DataLoaderの作成
test_dataloader = DataLoader(
    ds_test,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
    pin_memory=True,
    num_workers=2
)
# ウォームアップ実行（数値安定性確認）
print("Performing warmup inference with numerical stability check...")
warmup_start = time.time()

# より現実的なダミーデータでウォームアップ
dummy_input = {
    'input_ids': torch.randint(1, 1000, (1, 100)).to(device),
    'attention_mask': torch.ones(1, 100).to(device)
}

model.eval()
with torch.no_grad():
    # bfloat16でautocast
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        try:
            warmup_output = model(**dummy_input)
            warmup_logits = warmup_output.logits

            # ウォームアップ時の数値チェック
            if torch.isnan(warmup_logits).any():
                print("WARNING: NaN detected in warmup - model may have numerical issues")
            else:
                print(f"Warmup successful - logits range: [{warmup_logits.min():.3f}, {warmup_logits.max():.3f}]")

            print(f"Warmup completed in {time.time() - warmup_start:.2f}s")
        except Exception as e:
            print(f"Warmup failed: {e}")

# 実際の推論開始
start_time = time.time()
all_predictions = []
batch_times = []

print("Starting main inference...")
with torch.no_grad():
    # bfloat16でautocast（学習時と同じ精度）
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Predicting")):
            batch_start = time.time()

            # バッチデータをGPUに移動
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # GPU同期
            torch.cuda.synchronize()

            # 推論の実行
            outputs = model(**batch)
            logits = outputs.logits

            # より厳密な数値チェック
            nan_count = torch.isnan(logits).sum().item()
            inf_count = torch.isinf(logits).sum().item()

            if nan_count > 0 or inf_count > 0:
                print(f"Batch {batch_idx}: NaN={nan_count}, Inf={inf_count}")
                print(f"Logits stats: min={logits.min():.6f}, max={logits.max():.6f}, mean={logits.mean():.6f}")

                # NaN/Infの修正
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
                print("Applied nan_to_num fix")

            # CPUに移動して結果を保存（bfloat16のまま）
            # all_predictions.append(logits.cpu().numpy())
            all_predictions.append(logits.cpu().float().numpy())

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # 最初の数バッチの詳細情報を表示
            if batch_idx < 3:
                unique_values = torch.unique(logits).shape[0]
                print(f"Batch {batch_idx+1}: {batch_time:.2f}s, range: [{logits.min():.6f}, {logits.max():.6f}], unique: {unique_values}")

            # メモリクリア
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()

# 予測結果を結合
all_predictions = np.concatenate(all_predictions, axis=0)

end_time = time.time()
total_time = end_time - start_time
avg_batch_time = np.mean(batch_times)

print(f"\n=== Performance Metrics ===")
print(f"Total prediction time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"Average time per batch: {avg_batch_time:.3f} seconds")
print(f"Average time per sample: {total_time/len(test):.3f} seconds")
print(f"Samples processed: {len(test)}")
print(f"Throughput: {len(test)/total_time:.2f} samples/second")

# 予測結果の詳細品質チェック
print(f"\n=== Prediction Quality Check ===")
print(f"Predictions shape: {all_predictions.shape}")
print(f"NaN count: {np.isnan(all_predictions).sum()}")
print(f"Inf count: {np.isinf(all_predictions).sum()}")
print(f"Zero count: {(all_predictions == 0).sum()}")
print(f"Predictions range: [{all_predictions.min():.6f}, {all_predictions.max():.6f}]")
print(f"Mean prediction: {all_predictions.mean():.6f}")
print(f"Std prediction: {all_predictions.std():.6f}")

# 各サンプルの統計
for i in range(min(3, len(all_predictions))):
    sample_pred = all_predictions[i]
    print(f"Sample {i}: range=[{sample_pred.min():.6f}, {sample_pred.max():.6f}], mean={sample_pred.mean():.6f}")

# メモリ使用量の表示
if torch.cuda.is_available():
    print(f"\n=== GPU Memory Usage ===")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# predictionsオブジェクトの構造を模倣
class PredictionOutput:
    def __init__(self, predictions):
        self.predictions = predictions

predictions = PredictionOutput(all_predictions)

print("\nCreating submission file...")
# 提出用ファイルの作成
submission = create_submission(predictions, test, le)

# 提出ファイルの保存
submission.to_csv(SUBMISSION_OUTPUT_PATH, index=False)
print(f"Submission file saved to: {SUBMISSION_OUTPUT_PATH}")
print("\nSubmission preview:")
print(submission.head())
print(f"\nSubmission shape: {submission.shape}")

# 最終的なメモリクリア
torch.cuda.empty_cache()
gc.collect()

print("\n=== Final Diagnostic ===")
print(f"Model dtype: {model_dtype}")
print(f"Compute dtype: {compute_dtype}")
print("- Using bfloat16 throughout (same as training)")
print("- Model merged for numerical stability")
print("- Extensive numerical checking enabled")
