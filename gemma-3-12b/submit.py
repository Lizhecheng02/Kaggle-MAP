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
                    torch_dtype=torch.bfloat16,  # 全精度で読み込み（NaN対策）
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
                # NaN対策のため、半精度変換をスキップ

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


                # attention_maskを使用して最後の有効なトークンの位置を取得
                if attention_mask is not None:
                    # 各サンプルの最後の有効なトークンの位置を見つける
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    batch_size = hidden_states.shape[0]
                    # 各サンプルから正しい位置の隠れ状態を取得
                    pooled_output = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
                else:
                    # attention_maskがない場合は従来通り最後の位置を使用
                    pooled_output = hidden_states[:, -1, :]


                pooled_output = self.dropout(pooled_output)


                # classifierを同じデバイスに移動
                self.classifier = self.classifier.to(pooled_output.device)


                logits = self.classifier(pooled_output)


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
                    # After loading classifier weights from safetensors
                    model.classifier.weight.data = model.classifier.weight.data.to(torch.bfloat16)
                    if model.classifier.bias is not None:
                        model.classifier.bias.data = model.classifier.bias.data.to(torch.bfloat16)

                    # classifierの重みを設定（float32で保持）
                    #model.classifier.weight.data = f.get_tensor(classifier_weight_key).to(model.classifier.weight.device).float()
                    #if classifier_bias_key in f.keys():
                    #    model.classifier.bias.data = f.get_tensor(classifier_bias_key).to(model.classifier.bias.device).float()
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

    # Trainerを使わず直接推論を実行（device_map="auto"との競合を避けるため）
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import numpy as np

    # DataLoaderの作成
    dataloader = DataLoader(
        ds_test,
        batch_size=4,  # T4 GPU 2枚で最適化
        collate_fn=data_collator,
        shuffle=False
    )

    # 推論の実行
    all_predictions = []
    model.eval()


    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            # バッチデータをモデルのデバイスに移動
            # device_map="auto"を使用している場合、model.deviceが最初のレイヤーのデバイス
            device = next(model.parameters()).device

            # 入力データをデバイスに移動
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            #if attention_mask is not None:
            #    attention_mask = attention_mask.to(torch.bfloat16)

            # 推論実行
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # logitsを取得
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits

            logits = logits.float()


            # CPUに移動してnumpy配列に変換
            predictions = logits.cpu().numpy()
            all_predictions.append(predictions)

    # 予測結果を結合
    all_predictions = np.vstack(all_predictions)


    # Trainer.predict()の出力形式に合わせる
    from transformers.trainer_utils import EvalPrediction
    predictions = EvalPrediction(predictions=all_predictions, label_ids=None)

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
