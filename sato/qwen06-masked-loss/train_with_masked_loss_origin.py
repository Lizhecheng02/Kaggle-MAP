"""
Qwen-3-0.6B モデルトレーニングスクリプト - QuestionIDベースのマスク付き損失関数バージョン
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoConfig
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoModel
import wandb
from transformers import EarlyStoppingCallback, TrainerCallback
import pickle
from collections import defaultdict

# カスタムモジュールのインポート（選択肢付きバージョン）
from config import *
from utils_with_choices import prepare_correct_answers, prepare_answer_choices, format_input, compute_map3
from data_collator import DataCollatorWithPadding


class SaveBestMap3Callback(TrainerCallback):
    """eval_map@3が最高値を更新した際にモデルを保存するコールバック"""
    def __init__(self, save_dir, tokenizer):
        self.save_dir = save_dir
        self.tokenizer = tokenizer
        self.best_map3 = 0.0

    def on_evaluate(self, args, state, control, metrics, model=None, **kwargs):
        current_map3 = metrics.get('eval_map@3', 0.0)

        if current_map3 > self.best_map3:
            self.best_map3 = current_map3

            # 専用ディレクトリに保存
            best_map3_path = os.path.join(self.save_dir, 'best_map3')
            os.makedirs(best_map3_path, exist_ok=True)

            # LoRAアダプターのみを保存
            model.save_pretrained(best_map3_path)
            self.tokenizer.save_pretrained(best_map3_path)

            print(f"\n新しいベストMAP@3スコア: {current_map3:.4f} - モデルを {best_map3_path} に保存しました")

        return control


class Qwen2ForSequenceClassificationWithMaskedLoss(nn.Module):
    """Qwen2モデルを分類タスク用にカスタマイズ - マスク付き損失関数版"""
    def __init__(self, model_name, num_labels, question_label_map=None, 
                 question_true_label_map=None, question_false_label_map=None):
        super().__init__()
        from transformers import AutoModel
        self.qwen = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.qwen.config.hidden_size, num_labels)
        self.num_labels = num_labels
        
        # PEFTライブラリとの互換性のためconfigを追加
        self.config = self.qwen.config
        self.config.num_labels = num_labels
        
        # QuestionIdごとの有効ラベルマップ（3種類）
        self.question_label_map = question_label_map
        self.question_true_label_map = question_true_label_map  # True_カテゴリのラベル
        self.question_false_label_map = question_false_label_map  # False_カテゴリのラベル
        
        # マスク値（無効なラベルに適用する大きな負の値）
        self.mask_value = -1e10

    def forward(self, input_ids, attention_mask=None, labels=None, question_ids=None, is_correct=None, **kwargs):
        # Transformersが渡す追加の引数（inputs_embeds等）をkwargsで受け取る
        # 基本のqwenモデルに必要な引数のみを渡す
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        # 最後のトークンの隠れ状態を使用
        pooled_output = outputs.last_hidden_state[:, -1, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # QuestionIdベースのマスクを適用（is_correctも考慮）
        if question_ids is not None and self.question_label_map is not None:
            masked_logits = self.apply_question_mask(logits, question_ids, is_correct)
            # デバッグ: 最初のバッチで確認
            # if torch.rand(1).item() < 0.001:  # 0.1%の確率でデバッグ出力
                # print(f"[MASK DEBUG] Original logits range: [{logits.min():.3f}, {logits.max():.3f}]")
                # print(f"[MASK DEBUG] Masked logits range: [{masked_logits.min():.3f}, {masked_logits.max():.3f}]")
                # print(f"[MASK DEBUG] Question IDs: {question_ids[:3].tolist()}")
        else:
            masked_logits = logits

        loss = None
        
        if labels is not None:
            # マスクされたlogitsで損失を計算
            loss_fct = nn.CrossEntropyLoss()
            # ラベルの形状をチェック
            if labels.dim() > 1:
                labels = labels.view(-1)
            if masked_logits.dim() == 3:
                masked_logits = masked_logits.view(-1, self.num_labels)
            
            # 損失を計算
            try:
                loss = loss_fct(masked_logits, labels)
                
                # 最初の数回でデバッグ情報を表示
                # if torch.rand(1).item() < 0.001:  # 0.1%の確率でデバッグ出力
                    # print(f"[LOSS DEBUG] Computed loss: {loss.item():.6f}")
                    # print(f"[LOSS DEBUG] Labels: {labels[:5].tolist()}")
                    # print(f"[LOSS DEBUG] Masked logits shape: {masked_logits.shape}")
                
                # 損失がNaNまたは無限大でないことを確認
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss detected: {loss}")
                    print(f"masked_logits stats: min={masked_logits.min()}, max={masked_logits.max()}, mean={masked_logits.mean()}")
                    print(f"labels stats: min={labels.min()}, max={labels.max()}")
                    # NaN/Infの場合は大きな損失値を設定
                    loss = torch.tensor(100.0, requires_grad=True, device=masked_logits.device)
                    
            except Exception as e:
                print(f"Error computing loss: {e}")
                print(f"masked_logits shape: {masked_logits.shape}, labels shape: {labels.shape}")
                # エラーが発生した場合は大きな損失値を設定
                loss = torch.tensor(100.0, requires_grad=True, device=masked_logits.device)

        # Accelerateライブラリ互換のため、SequenceClassifierOutputを使用
        return SequenceClassifierOutput(loss=loss, logits=masked_logits)
    
    def apply_question_mask(self, logits, question_ids, is_correct=None):
        """QuestionIdごとに無効なラベルをマスクする（is_correctも考慮、最適化版）"""
        batch_size = logits.size(0)
        
        # マスクを作成（初期値は全てマスク）
        mask = torch.full_like(logits, self.mask_value)
        
        for i in range(batch_size):
            q_id = question_ids[i].item() if torch.is_tensor(question_ids[i]) else question_ids[i]
            
            # 有効なラベルを決定（事前計算されたマッピングを使用）
            valid_labels = []
            
            if is_correct is not None:
                is_correct_val = is_correct[i].item() if torch.is_tensor(is_correct[i]) else is_correct[i]
                
                if is_correct_val == 1:
                    # 正解の場合：True_カテゴリのラベルのみ有効
                    if q_id in self.question_true_label_map:
                        valid_labels = self.question_true_label_map[q_id]
                else:
                    # 不正解の場合：False_カテゴリのラベルのみ有効
                    if q_id in self.question_false_label_map:
                        valid_labels = self.question_false_label_map[q_id]
            else:
                # is_correctがない場合：全ての有効なラベル
                if q_id in self.question_label_map:
                    valid_labels = self.question_label_map[q_id]
            
            # 有効なラベルのマスクを解除
            for label_idx in valid_labels:
                mask[i, label_idx] = 0
        
        # マスクを適用（無効なラベルには大きな負の値を加算）
        masked_logits = logits + mask
        
        return masked_logits


def create_question_label_mapping(train_df, label_encoder=None):
    """QuestionIdごとの有効なラベル（誤概念）のマッピングを作成（True_/False_別）"""
    question_label_map = defaultdict(set)
    question_true_label_map = defaultdict(set)  # True_カテゴリのラベル
    question_false_label_map = defaultdict(set)  # False_カテゴリのラベル
    
    for _, row in train_df.iterrows():
        question_id = row['QuestionId']
        label = row['label']
        question_label_map[question_id].add(label)
        
        # label_encoderがある場合、True_/False_別のマッピングも作成
        if label_encoder is not None:
            label_name = label_encoder.inverse_transform([label])[0]
            category = label_name.split(':')[0]
            
            if category.startswith('True_'):
                question_true_label_map[question_id].add(label)
            elif category.startswith('False_'):
                question_false_label_map[question_id].add(label)
    
    # setをlistに変換
    question_label_map = {q_id: list(labels) for q_id, labels in question_label_map.items()}
    question_true_label_map = {q_id: list(labels) for q_id, labels in question_true_label_map.items()}
    question_false_label_map = {q_id: list(labels) for q_id, labels in question_false_label_map.items()}
    
    # 統計情報を表示
    label_counts = [len(labels) for labels in question_label_map.values()]
    true_counts = [len(labels) for labels in question_true_label_map.values() if labels]
    false_counts = [len(labels) for labels in question_false_label_map.values() if labels]
    
    print(f"\n=== QuestionId-Label Mapping Statistics ===")
    print(f"Total unique questions: {len(question_label_map)}")
    print(f"Average labels per question: {np.mean(label_counts):.2f}")
    print(f"Min labels per question: {np.min(label_counts)}")
    print(f"Max labels per question: {np.max(label_counts)}")
    print(f"Median labels per question: {np.median(label_counts):.1f}")
    
    if label_encoder is not None:
        print(f"\n=== True_/False_ Category Statistics ===")
        print(f"Questions with True_ labels: {len(question_true_label_map)}")
        print(f"Questions with False_ labels: {len(question_false_label_map)}")
        if true_counts:
            print(f"Average True_ labels per question: {np.mean(true_counts):.2f}")
        if false_counts:
            print(f"Average False_ labels per question: {np.mean(false_counts):.2f}")
    
    # ヒストグラムを作成
    plt.figure(figsize=(10, 6))
    plt.hist(label_counts, bins=30, edgecolor='black')
    plt.xlabel('Number of labels per question')
    plt.ylabel('Number of questions')
    plt.title('Distribution of Labels per QuestionId')
    plt.grid(True, alpha=0.3)
    plt.savefig('question_label_distribution.png')
    plt.close()
    
    # 3つのマッピングを返す
    return dict(question_label_map), dict(question_true_label_map), dict(question_false_label_map)


def tokenize_dataset_with_question_id(dataset, tokenizer, max_len):
    """データセットのトークナイズ（QuestionIdとis_correct付き）"""
    def tokenize(batch):
        # パディングはDataCollatorで行うため、ここではトークナイズのみ
        tokenized = tokenizer(
            batch['text'],
            padding=False,  # パディングはDataCollatorに任せる
            truncation=True,
            max_length=max_len,
            return_tensors=None  # map時は'None'を使用
        )
        # QuestionIdとis_correctをそのまま保持
        tokenized['question_ids'] = batch['QuestionId']
        tokenized['is_correct'] = batch['is_correct']
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=['text', 'QuestionId', 'is_correct']  # textとQuestionIdとis_correctを削除、labelは保持
    )
    
    return tokenized_dataset


class DataCollatorWithQuestionId:
    """QuestionIdを含むカスタムデータコレーター"""
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features):
        # デバッグ: 入力データの構造を確認
        # print(f"[COLLATOR DEBUG] Features keys: {list(features[0].keys())}")
        # print(f"[COLLATOR DEBUG] First feature sample: {features[0]}")
        
        # バッチの最大長を取得
        max_length = max(len(feature["input_ids"]) for feature in features)
        
        # パディング
        batch = {}
        for key in features[0].keys():
            if key in ["label", "labels"]:
                # ラベルはパディング不要（labelをlabelsに変換）
                labels = [f[key] for f in features]
                print(f"[COLLATOR DEBUG] Processing labels: {labels[:5]}...")
                batch["labels"] = torch.tensor(labels, dtype=torch.long)
            elif key in ["input_ids", "attention_mask"]:
                # input_idsとattention_maskをパディング
                padded = []
                for feature in features:
                    # tensorをlistに変換
                    if torch.is_tensor(feature[key]):
                        feature_list = feature[key].tolist()
                    else:
                        feature_list = feature[key]
                    
                    remainder = [self.tokenizer.pad_token_id if key == "input_ids" else 0] * (max_length - len(feature_list))
                    padded_feature = feature_list + remainder
                    padded.append(padded_feature)
                batch[key] = torch.tensor(padded, dtype=torch.long)
            elif key == "question_ids":
                # question_idsはパディング不要
                batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
            elif key == "is_correct":
                # is_correctはパディング不要（float型で保持）
                batch[key] = torch.tensor([f[key] for f in features], dtype=torch.float)
        
        return batch


def main():
    """メイントレーニング関数"""

    # WandBの初期化
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME + "_masked_loss",
            entity=WANDB_ENTITY,
            config={
                "model_name": MODEL_NAME,
                "epochs": EPOCHS,
                "max_len": MAX_LEN,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE if USE_EARLY_STOPPING else None,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "lora_target_modules": LORA_TARGET_MODULES,
                "lora_dropout": LORA_DROPOUT,
                "lora_bias": LORA_BIAS,
                "with_choices": True,
                "masked_loss": True,  # マスク付き損失関数を使用
            }
        )

    # GPU設定
    if CUDA_VISIBLE_DEVICES is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
        print(f"Using CUDA device(s): {CUDA_VISIBLE_DEVICES}")

    # 出力ディレクトリの作成
    output_dir_masked = OUTPUT_DIR + "_masked_loss"
    os.makedirs(output_dir_masked, exist_ok=True)

    # --- データの読み込みと前処理 ---
    print("Loading and preprocessing training data...")
    le = LabelEncoder()
    train = pd.read_csv(TRAIN_DATA_PATH)
    
    # --- QuestionId 32835のQuestionTextを更新 ---
    print("Updating QuestionId 32835...")
    new_question_text = "Which number is the greatest? Options: 6.0000 6.2 6.079 6.0001"
    mask_32835 = train['QuestionId'] == 32835
    update_count = mask_32835.sum()
    
    if update_count > 0:
        original_text = train[mask_32835]['QuestionText'].iloc[0]
        print(f"Found {update_count} rows with QuestionId 32835")
        print(f"Original: {original_text[:80]}...")
        print(f"Updated to: {new_question_text}")
        train.loc[mask_32835, 'QuestionText'] = new_question_text
    else:
        print("No rows found with QuestionId 32835")
    
    # フィルタリングを行わず全データを使用
    print(f"Using all data without filtering: {train.shape[0]} rows")
    
    train.Misconception = train.Misconception.fillna('NA')
    train['target'] = train.Category + ":" + train.Misconception
    train['label'] = le.fit_transform(train['target'])
    n_classes = len(le.classes_)
    print(f"Train shape: {train.shape} with {n_classes} target classes")

    # --- 特徴量エンジニアリング ---
    print("Performing feature engineering...")
    correct = prepare_correct_answers(train)
    train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    train.is_correct = train.is_correct.fillna(0)

    # --- 選択肢データの準備 ---
    print("Preparing answer choices for each question...")
    choices = prepare_answer_choices(train)
    train = train.merge(choices[['QuestionId', 'answer_choices_str']], on='QuestionId', how='left')
    
    # --- MC_Answerを選択肢ラベルに変換 ---
    print("Converting MC_Answer to choice labels...")
    def get_choice_label(row):
        question_id = row['QuestionId']
        mc_answer = row['MC_Answer']
        # 該当するchoice_mappingを取得
        choice_mapping = choices[choices['QuestionId'] == question_id]['choice_mapping'].iloc[0]
        return choice_mapping.get(mc_answer, mc_answer)  # マッピングがない場合は元の値
    
    train['choice_label'] = train.apply(get_choice_label, axis=1)

    # --- 入力テキストのフォーマット ---
    print("Formatting input text with answer choices...")
    train['text'] = train.apply(format_input, axis=1)
    print("Example prompt for our LLM with choices:")
    print(train.text.values[0])

    # --- QuestionId-Labelマッピングの作成 ---
    print("\nCreating QuestionId-Label mapping for masked loss...")
    question_label_map, question_true_label_map, question_false_label_map = create_question_label_mapping(train, le)
    
    # マッピングを保存
    mapping_path = f"{output_dir_masked}/question_label_mapping.pkl"
    with open(mapping_path, 'wb') as f:
        pickle.dump({
            'question_label_map': question_label_map,
            'question_true_label_map': question_true_label_map,
            'question_false_label_map': question_false_label_map
        }, f)
    print(f"Question-Label mapping (3 types) saved to: {mapping_path}")

    # --- トークナイザーの初期化 ---
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # パディングトークンの設定
    # Qwen3モデルの場合、特別なトークンIDを使用
    if tokenizer.pad_token is None:
        # 語彙内の安全なトークンIDを使用
        # Qwenモデルでは、0番のトークンがUNKNOWNトークンとして使われることが多い
        tokenizer.pad_token_id = 0
        tokenizer.pad_token = tokenizer.decode([0])

    # --- トークン長の分析 ---
    print("Analyzing token lengths...")
    lengths = [len(tokenizer.encode(t, truncation=False)) for t in train['text']]
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50)
    plt.title("Token Length Distribution (With Choices and Masked Loss)")
    plt.xlabel("Number of tokens")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f'{output_dir_masked}/token_length_distribution_masked.png')
    plt.close()

    over_limit = (np.array(lengths) > MAX_LEN).sum()
    print(f"There are {over_limit} train sample(s) with more than {MAX_LEN} tokens")
    
    # 選択肢が追加されたことによるトークン長の増加を表示
    avg_length = np.mean(lengths)
    max_length = np.max(lengths)
    print(f"Average token length: {avg_length:.1f}, Max token length: {max_length}")

    # --- データの分割 ---
    if VALIDATION_SPLIT > 0:
        print("Splitting data into train and validation sets...")
        train_df, val_df = train_test_split(train, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
        
        # QuestionIdとis_correctを含むカラムを選択
        COLS = ['text', 'label', 'QuestionId', 'is_correct']
        train_ds = Dataset.from_pandas(train_df[COLS])
        val_ds = Dataset.from_pandas(val_df[COLS])
        
        # --- データセットのトークナイズ ---
        print("Tokenizing datasets with QuestionIds...")
        train_ds = tokenize_dataset_with_question_id(train_ds, tokenizer, MAX_LEN)
        val_ds = tokenize_dataset_with_question_id(val_ds, tokenizer, MAX_LEN)
    else:
        print("Using all data for training (no validation split)...")
        train_df = train
        val_df = None
        
        # QuestionIdとis_correctを含むカラムを選択
        COLS = ['text', 'label', 'QuestionId', 'is_correct']
        train_ds = Dataset.from_pandas(train_df[COLS])
        val_ds = None
        
        # --- データセットのトークナイズ ---
        print("Tokenizing training dataset with QuestionIds...")
        train_ds = tokenize_dataset_with_question_id(train_ds, tokenizer, MAX_LEN)

    # --- Label Encoderの保存 ---
    label_encoder_path = f"{output_dir_masked}/label_encoder.joblib"
    print(f"Saving label encoder to: {label_encoder_path}")
    joblib.dump(le, label_encoder_path)

    # --- モデルの初期化 ---
    print("Initializing model with masked loss...")
    # カスタムクラスを直接使用（マスク付き損失関数のため、3つのマッピングを渡す）
    model = Qwen2ForSequenceClassificationWithMaskedLoss(
        MODEL_NAME, 
        n_classes, 
        question_label_map, 
        question_true_label_map, 
        question_false_label_map
    )
    
    # パディングトークンIDを設定
    if hasattr(model.config, 'pad_token_id'):
        model.config.pad_token_id = tokenizer.pad_token_id

    # --- LoRAアダプターの設定 ---
    print("Configuring LoRA adapter...")
    lora_config = LoraConfig(
        r=LORA_RANK,  # LoRAのランク
        lora_alpha=LORA_ALPHA,  # LoRAのスケーリングパラメータ
        target_modules=LORA_TARGET_MODULES,  # 対象モジュール
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        task_type=TaskType.SEQ_CLS,
    )

    # PEFTモデルの作成
    model = get_peft_model(model, lora_config)
    print("Number of trainable parameters:")
    model.print_trainable_parameters()

    # シングルGPUに設定
    if torch.cuda.is_available():
        model = model.cuda()

    # --- トレーニング引数の設定 ---
    training_args = TrainingArguments(
        output_dir=output_dir_masked,
        do_train=True,
        do_eval=val_ds is not None,  # validationデータがある場合のみevaluation実行
        eval_strategy="steps" if val_ds is not None else "no",
        save_strategy="steps" if val_ds is not None else "epoch",
        eval_steps=EVAL_STEPS if val_ds is not None else None,
        save_steps=SAVE_STEPS if val_ds is not None else None,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_dir=f"{output_dir_masked}/logs",
        logging_steps=LOGGING_STEPS,
        metric_for_best_model="map@3" if val_ds is not None else None,
        greater_is_better=True,
        load_best_model_at_end=val_ds is not None,
        report_to="wandb" if USE_WANDB else "none",
        bf16=True,
        gradient_checkpointing=False,  # インデックスエラーのため一時的に無効化
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # メモリ効率向上のため追加
        remove_unused_columns=False,  # カラムを削除しない
        lr_scheduler_type="cosine",  # コサインスケジューラーを使用
        warmup_ratio=0.0,  # ウォームアップを無効化
        save_total_limit=2,
    )

    # --- トレーナーのセットアップとトレーニング ---
    print("Setting up trainer...")

    # エポックあたりのステップ数を計算
    steps_per_epoch = len(train_ds) // (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)  # gradient_accumulation_stepsを考慮
    total_steps = steps_per_epoch * EPOCHS
    print(f"\nDataset statistics:")
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds) if val_ds is not None else 0}")
    print(f"Batch size: {TRAIN_BATCH_SIZE} (with gradient accumulation: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")
    if val_ds is not None:
        print(f"Evaluation interval: every {EVAL_STEPS} steps (~{EVAL_STEPS/steps_per_epoch:.2f} epochs)")
        print(f"Early stopping after {EARLY_STOPPING_PATIENCE} evaluations without improvement")
    else:
        print("No validation - training without evaluation")

    # カスタムデータコレーターを使用（QuestionId付き）
    data_collator = DataCollatorWithQuestionId(tokenizer=tokenizer, max_length=MAX_LEN)

    # コールバックの設定
    callbacks = []

    # SaveBestMap3Callbackを追加（validationがある場合のみ）
    if val_ds is not None:
        save_best_callback = SaveBestMap3Callback(save_dir=output_dir_masked, tokenizer=tokenizer)
        callbacks.append(save_best_callback)
        print(f"SaveBestMap3Callback enabled - モデルは {output_dir_masked}/best_map3 に保存されます")

        if USE_EARLY_STOPPING:
            # EARLY_STOPPING_PATIENCEは評価回数として直接使用
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD
            )
            callbacks.append(early_stopping_callback)
            print(f"Early stopping enabled:")
            print(f"  - Patience (evaluations without improvement): {EARLY_STOPPING_PATIENCE}")
            print(f"  - Threshold: {EARLY_STOPPING_THRESHOLD}")
    else:
        print("No validation - callbacks disabled")

    # デバッグ: compute_map3関数の確認
    # print(f"[DEBUG] compute_map3 function: {compute_map3}")
    # print(f"[DEBUG] val_ds is not None: {val_ds is not None}")
    compute_metrics_func = compute_map3 if val_ds is not None else None
    print(f"[DEBUG] compute_metrics will be set to: {compute_metrics_func}")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_func,
        callbacks=callbacks,
    )
    
    # Trainerが初期化された後にcompute_metricsが正しく設定されているか確認
    print(f"[DEBUG] Trainer.compute_metrics: {trainer.compute_metrics}")
    if hasattr(trainer, 'compute_metrics') and trainer.compute_metrics:
        print("[DEBUG] compute_metrics is properly set in Trainer")
    else:
        print("[DEBUG] WARNING: compute_metrics is NOT set in Trainer")

    print("\nStarting training with masked loss...")
    print("Note: Invalid labels for each QuestionId will be masked during training.")
    trainer.train()

    # --- 最終的なMAP@3スコアを表示 ---
    if val_ds is not None:
        print("\nEvaluating on validation set...")
        eval_results = trainer.evaluate()
        print(f"\nValidation MAP@3: {eval_results.get('eval_map@3', 'N/A'):.4f}")

    # --- モデルの保存 ---
    best_model_path = f"{output_dir_masked}/best"
    print("\nSaving model...")
    # LoRAアダプターのみを保存
    model.save_pretrained(best_model_path)
    # トークナイザーも保存
    tokenizer.save_pretrained(best_model_path)

    print("Training completed successfully!")
    print(f"Model saved to: {best_model_path}")
    print(f"Label encoder saved to: {label_encoder_path}")
    print(f"Question-Label mapping saved to: {mapping_path}")

    # WandBの終了
    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()