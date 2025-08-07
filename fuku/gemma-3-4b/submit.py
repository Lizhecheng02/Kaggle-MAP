"""
Gemma-3-1B モデル推論スクリプト - 提出用予測ファイルの生成
"""

import pandas as pd
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import joblib
import torch
import gc
import os
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


# Gemmaモデル用のカスタムクラスが必要な場合のためにインポート
try:
    from train import GemmaForSequenceClassification
except ImportError:
    GemmaForSequenceClassification = None


def main():
    """メイン推論関数"""

    # GPUメモリをクリア
    torch.cuda.empty_cache()
    gc.collect()

    # 複数GPU対応 - device_map="auto"が自動的に処理するため、特定のGPU指定は不要
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    # メモリ効率化のための設定
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Loading label encoder...")
    # ラベルエンコーダーの読み込み
    le = joblib.load(LABEL_ENCODER_PATH)
    n_classes = len(le.classes_)

    print("Loading trained model and tokenizer...")

    if PEFT_AVAILABLE:
        # LoRAアダプターを使用する場合
        print(f"Loading fine-tuned LoRA model from: {BEST_MODEL_PATH}")
        print(f"Loading base model from: {MODEL_NAME}")

        # カスタムGemmaモデルクラスを修正して、効率的に読み込む
        import torch.nn as nn
        from transformers import AutoModel

        class GemmaForSequenceClassificationOptimized(nn.Module):
            def __init__(self, model_name, num_labels):
                super().__init__()
                # device_mapを使って自動的に複数GPUに分散
                # Gemma3ForConditionalGenerationモデルを読み込む
                self.gemma = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,  # fp16で読み込み
                    device_map="auto",  # 自動的に複数GPUに分散
                    low_cpu_mem_usage=True  # CPUメモリ使用量を削減
                )
                self.config = self.gemma.config
                self.dropout = nn.Dropout(0.1)
                # text_configのhidden_sizeを使用
                hidden_size = self.config.text_config.hidden_size
                self.classifier = nn.Linear(hidden_size, num_labels)
                # 分類器の重みを正しく初期化
                nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
                if self.classifier.bias is not None:
                    nn.init.zeros_(self.classifier.bias)
                # 分類器もfp16に変換
                self.classifier = self.classifier.half()

            def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None, **kwargs):
                if input_ids is None and inputs_embeds is None:
                    raise ValueError("You have to specify either input_ids or inputs_embeds")

                # Gemma3ForConditionalGenerationモデルの出力を取得
                outputs = self.gemma(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=True
                )

                # 言語モデルの隠れ状態を取得
                # Gemma3ForConditionalGenerationの場合、language_modelの出力を使用
                if hasattr(outputs, 'language_model_outputs'):
                    hidden_states = outputs.language_model_outputs.hidden_states[-1]
                elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hidden_states = outputs.hidden_states[-1]
                else:
                    # 最後の隠れ状態を取得
                    hidden_states = outputs.last_hidden_state

                # デバッグ：hidden_statesの値を確認
                print(f"Debug - Hidden states has NaN: {torch.isnan(hidden_states).any().item()}")
                print(f"Debug - Hidden states shape: {hidden_states.shape}")

                # attention_maskを使用して最後の有効なトークンの位置を取得
                if attention_mask is not None:
                    # 各サンプルの最後の有効なトークンの位置を見つける
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    batch_size = hidden_states.shape[0]
                    print(f"Debug - Sequence lengths: {sequence_lengths}")
                    # 各サンプルから正しい位置の隠れ状態を取得
                    pooled_output = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
                else:
                    # attention_maskがない場合は従来通り最後の位置を使用
                    pooled_output = hidden_states[:, -1, :]

                # デバッグ：pooled_outputの値を確認
                print(f"Debug - Pooled output shape: {pooled_output.shape}, min/max: {pooled_output.min().item():.4f}/{pooled_output.max().item():.4f}")
                print(f"Debug - Pooled output has NaN: {torch.isnan(pooled_output).any().item()}")

                pooled_output = self.dropout(pooled_output)

                # デバッグ：dropout後の値を確認
                print(f"Debug - After dropout has NaN: {torch.isnan(pooled_output).any().item()}")

                # classifierを同じデバイスに移動
                self.classifier = self.classifier.to(pooled_output.device)

                # デバッグ：classifierの重みを確認
                print(f"Debug - Classifier weight has NaN: {torch.isnan(self.classifier.weight).any().item()}")
                print(f"Debug - Classifier bias has NaN: {torch.isnan(self.classifier.bias).any().item() if self.classifier.bias is not None else 'No bias'}")

                logits = self.classifier(pooled_output)

                # デバッグ：logits直後の値を確認
                print(f"Debug - Logits has NaN: {torch.isnan(logits).any().item()}")

                loss = None
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

                if loss is not None:
                    return {'loss': loss, 'logits': logits}
                else:
                    return {'logits': logits}

        # PeftConfigから正しいベースモデル名を取得
        peft_config = PeftConfig.from_pretrained(BEST_MODEL_PATH)
        print(f"LoRA adapter expects base model: {peft_config.base_model_name_or_path}")

        # 最適化されたモデルを使用
        model = GemmaForSequenceClassificationOptimized(MODEL_NAME, num_labels=n_classes)

        # LoRAアダプターを適用前に、classifierの重みを保存された重みで置き換える必要がある
        # まず、保存されたチェックポイントから分類器の重みを読み込む
        import os
        from safetensors import safe_open

        # safetensorsファイルから重みを読み込む
        safetensors_path = os.path.join(BEST_MODEL_PATH, "adapter_model.safetensors")
        if os.path.exists(safetensors_path):
            with safe_open(safetensors_path, framework='pt') as f:
                # classifierの重みを探す
                classifier_weight_key = "base_model.model.classifier.weight"
                classifier_bias_key = "base_model.model.classifier.bias"

                if classifier_weight_key in f.keys():
                    print(f"Loading classifier weights from safetensors")
                    # classifierの重みを設定（fp16で保持）
                    model.classifier.weight.data = f.get_tensor(classifier_weight_key).to(model.classifier.weight.device).half()
                    if classifier_bias_key in f.keys():
                        model.classifier.bias.data = f.get_tensor(classifier_bias_key).to(model.classifier.bias.device).half()
                    print(f"Classifier weights loaded successfully")

        # LoRAアダプターを適用 - is_trainable=Falseを追加
        model = PeftModel.from_pretrained(
            model,
            BEST_MODEL_PATH,
            is_trainable=False  # 推論モードであることを明示
        )

        # device_map="auto"を使用しているため、DataParallelは不要
        print(f"Model loaded with automatic device mapping across {torch.cuda.device_count()} GPUs")

        # モデルを評価モードに設定
        model.eval()

        # デバッグ：LoRAアダプターが正しく読み込まれているか確認
        print("\nDebug - Checking LoRA adapter loading:")
        lora_params_found = False
        lora_weight_sum = 0.0
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                lora_params_found = True
                # メタテンソルの場合はスキップ
                if not param.is_meta:
                    lora_weight_sum += param.abs().sum().item()
                print(f"  Found LoRA layer: {name}, shape: {param.shape}, requires_grad: {param.requires_grad}")
                # 最初のLoRA層の重みの一部を表示
                if 'lora_B' in name and param.numel() > 0 and not param.is_meta:
                    print(f"    First few weights: {param.data.flatten()[:5].cpu().numpy()}")

        if not lora_params_found:
            print("  WARNING: No LoRA parameters found!")
        else:
            print(f"  Total sum of absolute LoRA weights: {lora_weight_sum}")
            if lora_weight_sum == 0:
                print("  WARNING: All LoRA weights are zero!")

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

    # デバッグ：最初の3つのサンプルの入力テキストを表示
    print("\nDebug - Sample input texts:")
    for i in range(min(3, len(test))):
        print(f"\nSample {i}:")
        print(f"QuestionId: {test.iloc[i]['QuestionId']}")
        print(f"MC_Answer: {test.iloc[i]['MC_Answer']}")
        print(f"is_correct: {test.iloc[i]['is_correct']}")
        print(f"Text (first 200 chars): {test.iloc[i]['text'][:200]}...")

    print("Tokenizing test data...")
    # テストデータのトークナイズ
    ds_test = Dataset.from_pandas(test[['text']])
    ds_test = tokenize_dataset(ds_test, tokenizer, MAX_LEN)

    # パディングのためのデータコラレータの設定
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Running inference...")

    # Trainerを使わず直接推論を実行（device_map="auto"との競合を避けるため）
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import numpy as np

    # DataLoaderの作成
    dataloader = DataLoader(
        ds_test,
        batch_size=1,  # メモリ節約のため1に固定
        collate_fn=data_collator,
        shuffle=False
    )

    # 推論の実行
    all_predictions = []
    model.eval()

    # デバッグ用：最初の5つのサンプルの入力とlogitsを確認
    debug_counter = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            # バッチデータをモデルのデバイスに移動
            # device_map="auto"を使用している場合、model.deviceが最初のレイヤーのデバイス
            device = next(model.parameters()).device

            # 入力データをデバイスに移動
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # 推論実行
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # logitsを取得
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits

            # デバッグ：最初の5つのサンプルのlogitsを表示
            if debug_counter < 5:
                print(f"\nDebug Sample {debug_counter + 1}:")
                print(f"Input shape: {input_ids.shape}")
                print(f"Attention mask sum: {attention_mask.sum().item() if attention_mask is not None else 'None'}")
                print(f"Logits shape: {logits.shape}")
                print(f"Logits (first 10 values): {logits[0][:10].cpu().numpy()}")
                print(f"Logits min/max: {logits.min().item():.4f} / {logits.max().item():.4f}")
                # Top-3予測を表示
                probs = torch.nn.functional.softmax(logits, dim=-1)
                top3_probs, top3_indices = torch.topk(probs[0], k=3)
                print(f"Top-3 predictions: {top3_indices.cpu().numpy()} with probs: {top3_probs.cpu().numpy()}")
                debug_counter += 1

            # CPUに移動してnumpy配列に変換
            predictions = logits.cpu().numpy()
            all_predictions.append(predictions)

    # 予測結果を結合
    all_predictions = np.vstack(all_predictions)

    # デバッグ：予測結果の統計情報を表示
    print(f"\nDebug - All predictions shape: {all_predictions.shape}")
    print(f"Debug - Predictions unique values per position:")
    for i in range(min(5, all_predictions.shape[0])):
        unique_vals = np.unique(all_predictions[i])
        print(f"  Sample {i}: {len(unique_vals)} unique values, range: [{all_predictions[i].min():.4f}, {all_predictions[i].max():.4f}]")
        # 全体の分散も確認
        print(f"    Variance: {np.var(all_predictions[i]):.6f}, Std: {np.std(all_predictions[i]):.6f}")

    # 予測結果の類似度を確認
    print("\nDebug - Checking prediction similarity:")
    if all_predictions.shape[0] >= 2:
        for i in range(min(3, all_predictions.shape[0]-1)):
            similarity = np.corrcoef(all_predictions[i], all_predictions[i+1])[0,1]
            print(f"  Correlation between sample {i} and {i+1}: {similarity:.6f}")

    # Trainer.predict()の出力形式に合わせる
    from transformers.trainer_utils import EvalPrediction
    predictions = EvalPrediction(predictions=all_predictions, label_ids=None)

    print("Creating submission file...")
    # 提出用ファイルの作成
    submission = create_submission(predictions, test, le)

    # デバッグ：Top-3予測の分析
    print("\nDebug - Top-3 predictions analysis:")
    for i in range(min(5, len(submission))):
        print(f"  Row {i}: {submission.iloc[i]['Category:Misconception']}")

    # ファイルの保存
    submission.to_csv(SUBMISSION_OUTPUT_PATH, index=False)
    print(f"Submission file saved to: {SUBMISSION_OUTPUT_PATH}")
    print("\nSubmission preview:")
    print(submission.head())
    print(f"\nSubmission shape: {submission.shape}")


if __name__ == "__main__":
    main()
