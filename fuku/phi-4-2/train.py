"""
Phi-4 モデルトレーニングスクリプト
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
from datasets import Dataset
import joblib
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoModel
import wandb
from transformers import EarlyStoppingCallback, TrainerCallback
import gc

# カスタムモジュールのインポート
from config import *
from utils import prepare_correct_answers, format_input, tokenize_dataset, compute_map3
from models import HierPhi4ForSequenceClassification
from data_collator import DataCollatorWithPadding


class GradientCheckCallback(TrainerCallback):
    """最初のステップで勾配が正しく計算されているか確認するコールバック"""
    def __init__(self):
        self.checked = False

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not self.checked and state.global_step == 1:
            self.checked = True
            print("\n" + "="*60)
            print("Gradient Check (Step 1)")
            print("="*60)

            # trainable パラメータの勾配を確認
            params_with_grad = []
            params_without_grad = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        params_with_grad.append((name, grad_norm))
                    else:
                        params_without_grad.append(name)

            print(f"Parameters with gradients: {len(params_with_grad)}")
            if params_with_grad:
                print("Sample gradient norms:")
                for name, norm in params_with_grad[:5]:
                    print(f"  {name}: {norm:.6f}")

            if params_without_grad:
                print(f"\n⚠ Warning: {len(params_without_grad)} trainable parameters have no gradients!")
                print("Sample parameters without gradients:")
                for name in params_without_grad[:5]:
                    print(f"  {name}")
            else:
                print("✓ All trainable parameters have gradients!")

            print("="*60 + "\n")

        return control


class SaveBestMap3Callback(TrainerCallback):
    """eval_{METRIC_NAME} が最高値を更新した際にモデルを保存するコールバック"""
    def __init__(self, save_dir, tokenizer, metric_name: str = METRIC_NAME):
        self.save_dir = save_dir
        self.tokenizer = tokenizer
        self.best_map3 = 0.0
        self.metric_name = metric_name
        self.metric_key = f"eval_{self.metric_name}"

    def on_evaluate(self, args, state, control, metrics, model=None, **kwargs):
        # デバッグ: 初回のみ metrics のキー一覧を出力
        if not hasattr(self, "_printed_keys"):
            try:
                print(f"[Debug] on_evaluate metrics keys: {sorted(list(metrics.keys()))}")
            except Exception:
                pass
            self._printed_keys = True

        current_map3 = metrics.get(self.metric_key, None)
        current_step = state.global_step
        total_steps = state.max_steps if state.max_steps else "N/A"

        if current_map3 is None:
            print(f"\n[Step {current_step}/{total_steps}] 注意: metrics に '{self.metric_key}' が存在しません（compute_metrics未実行の可能性）。")
            current_map3 = 0.0
        print(f"[Step {current_step}/{total_steps}] 評価実行 - MAP@{MAP_K}スコア: {current_map3:.4f}")

        if current_map3 > self.best_map3:
            self.best_map3 = current_map3

            # 専用ディレクトリに保存
            best_map3_path = os.path.join(self.save_dir, 'best_map3')
            os.makedirs(best_map3_path, exist_ok=True)

            # LoRAアダプターのみを保存
            model.save_pretrained(best_map3_path)
            self.tokenizer.save_pretrained(best_map3_path)

            print(f"🎉 新しいベストMAP@{MAP_K}スコア更新: {current_map3:.4f} (Step {current_step}) - モデルを {best_map3_path} に保存しました")
        else:
            print(f"現在のベストMAP@{MAP_K}スコア: {self.best_map3:.4f} (変更なし)")

        return control


class Phi4ForSequenceClassification(nn.Module):
    """Phi-4モデルを分類タスク用にカスタマイズ"""
    def __init__(self, backbone, num_labels):
        """
        Args:
            backbone: 事前にロードされたPhi-4 base model
            num_labels: 分類クラス数
        """
        super().__init__()
        self.phi = backbone
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.phi.config.hidden_size, num_labels)

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
    """メイントレーニング関数"""

    # config.pyの内容を出力
    print("=" * 80)
    print("Configuration Settings (config.py):")
    print("=" * 80)
    with open('config.py', 'r', encoding='utf-8') as f:
        print(f.read())
    print("=" * 80)
    print()

    # WandBの初期化
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
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
                "use_dora": USE_DORA,
                "attention_implementation": ATTENTION_IMPLEMENTATION,
            }
        )

    # GPU設定
    if CUDA_VISIBLE_DEVICES is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
        print(f"Using CUDA device(s): {CUDA_VISIBLE_DEVICES}")

    # メモリキャッシュをクリア
    torch.cuda.empty_cache()
    gc.collect()

    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- データの読み込みと前処理 ---
    print("Loading and preprocessing training data...")
    le = LabelEncoder()
    train = pd.read_csv(TRAIN_DATA_PATH)
    train.Misconception = train.Misconception.fillna('NA')
    train['target'] = train.Category + ":" + train.Misconception
    train['label'] = le.fit_transform(train['target'])
    n_classes = len(le.classes_)
    print(f"Train shape: {train.shape} with {n_classes} target classes")

    # --- 階層エンコーダ（Category / Misconception:NA含む）---
    from sklearn.preprocessing import LabelEncoder as _LE
    le_cat = _LE()
    le_mc = _LE()
    train['label_cat'] = le_cat.fit_transform(train['Category'].astype(str))
    train['label_mc'] = le_mc.fit_transform(train['Misconception'].astype(str))

    # joint -> (cat_idx, mc_idx) のマッピングを作成
    classes_joint = list(le.classes_)
    cat_for_joint = []
    mc_for_joint = []
    for t in classes_joint:
        c, m = t.split(":", 1)
        cat_for_joint.append(c)
        mc_for_joint.append(m)
    joint_to_cat = torch.tensor(le_cat.transform(cat_for_joint), dtype=torch.long)
    joint_to_mc = torch.tensor(le_mc.transform(mc_for_joint), dtype=torch.long)

    # Category が *_Misconception かどうかのフラグ
    cat_is_misconc = pd.Series(le_cat.classes_).astype(str).str.endswith('_Misconception').values
    cat_is_misconc = torch.tensor(cat_is_misconc, dtype=torch.bool)

    # Misconception の NA index
    if 'NA' not in set(le_mc.classes_):
        raise RuntimeError("Misconception エンコーダに 'NA' が含まれていません。")
    mc_na_index = int(le_mc.transform(['NA'])[0])

    # --- 特徴量エンジニアリング ---
    print("Performing feature engineering...")
    correct = prepare_correct_answers(train)
    train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    train.is_correct = train.is_correct.fillna(0)

    # --- 入力テキストのフォーマット ---
    print("Formatting input text...")
    train['text'] = train.apply(format_input, axis=1)
    print("Example prompt for our LLM:")
    print(train.text.values[0])

    # --- トークナイザーの初期化 ---
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # パディングトークンの設定
    # Phi-4モデルの場合の設定
    if tokenizer.pad_token is None:
        # Phi-4では特別なパディングトークンを使用
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = 100257

    # --- トークン長の分析 ---
    print("Analyzing token lengths...")
    lengths = [len(tokenizer.encode(t, truncation=False)) for t in train['text']]
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50)
    plt.title("Token Length Distribution")
    plt.xlabel("Number of tokens")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f'{OUTPUT_DIR}/token_length_distribution.png')
    plt.close()

    over_limit = (np.array(lengths) > MAX_LEN).sum()
    print(f"There are {over_limit} train sample(s) with more than {MAX_LEN} tokens")

    # --- データの分割 ---
    print("Splitting data into train and validation sets...")
    train_df, val_df = train_test_split(train, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
    COLS = ['text','label','label_cat','label_mc']
    train_ds = Dataset.from_pandas(train_df[COLS])
    val_ds = Dataset.from_pandas(val_df[COLS])

    # --- データセットのトークナイズ ---
    print("Tokenizing datasets...")
    train_ds = tokenize_dataset(train_ds, tokenizer, MAX_LEN)
    val_ds = tokenize_dataset(val_ds, tokenizer, MAX_LEN)

    # --- Label Encoderの保存 ---
    print(f"Saving label encoder (joint) to: {LABEL_ENCODER_PATH}")
    joblib.dump(le, LABEL_ENCODER_PATH)
    # 階層メタの保存（互換性のため別ファイル）
    print(f"Saving hierarchical metadata to: {HIER_META_PATH}")
    hier_meta = {
        'le_cat': le_cat,
        'le_mc': le_mc,
        'joint_to_cat': joint_to_cat,
        'joint_to_mc': joint_to_mc,
        'cat_is_misconc': cat_is_misconc,
        'mc_na_index': mc_na_index,
        'joint_classes': classes_joint,
    }
    joblib.dump(hier_meta, HIER_META_PATH)

    # --- モデルの初期化 ---
    print("Initializing model...")
    print(f"Using attention implementation: {ATTENTION_IMPLEMENTATION}")

    # モデルロード時間の計測開始
    import time
    load_start_time = time.time()

    # ベース（LLM本体）を読み込む
    # device_map="auto"でモデルを直接GPUに効率的にロード
    base_model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",  # 修正: Noneから"auto"に変更し、効率的なGPUロードを実現
        torch_dtype=torch.bfloat16,  # BF16で読み込み
        low_cpu_mem_usage=True,
        attn_implementation=ATTENTION_IMPLEMENTATION
    )

    load_end_time = time.time()
    load_duration = load_end_time - load_start_time
    print(f"Model loaded in {load_duration:.2f} seconds")

    # GPUメモリ使用量を確認
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    # パディングトークンIDを設定
    base_model.config.pad_token_id = tokenizer.pad_token_id

    if HIERARCHICAL_TRAINING and COMPOSE_JOINT_FROM_HEADS:
        print("Using Hierarchical multi-task model (cat/mc heads + composed joint)")
        model_core = HierPhi4ForSequenceClassification(
            backbone=base_model,
            hidden_size=base_model.config.hidden_size,
            n_joint=n_classes,
            n_cat=len(le_cat.classes_),
            n_mc=len(le_mc.classes_),
            joint_to_cat=joint_to_cat,
            joint_to_mc=joint_to_mc,
            cat_is_misconc=cat_is_misconc,
            mc_na_index=mc_na_index,
            lambda_cat=LAMBDA_CAT,
            lambda_mc=LAMBDA_MC,
            lambda_constraint=LAMBDA_CONSTRAINT,
        )
    else:
        print("Fallback: using simple single-head classifier for joint labels")
        # 互換用のシンプルモデル（必要時のみ）
        # 修正: backboneを直接渡すことで、不要な再ロードを回避
        model_core = Phi4ForSequenceClassification(backbone=base_model, num_labels=n_classes)

    # --- LoRAアダプターの設定 ---
    print("Configuring LoRA adapter...")
    modules_to_save = None
    if HIERARCHICAL_TRAINING and COMPOSE_JOINT_FROM_HEADS:
        modules_to_save = ["fc_cat", "fc_mc"]

    lora_config = LoraConfig(
        r=LORA_RANK,  # LoRAのランク
        lora_alpha=LORA_ALPHA,  # LoRAのスケーリングパラメータ
        target_modules=LORA_TARGET_MODULES,  # 対象モジュール
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        task_type=TaskType.SEQ_CLS,
        use_dora=USE_DORA,  # DoRAの使用
        modules_to_save=modules_to_save,
    )

    # PEFTモデルの作成
    model = get_peft_model(model_core, lora_config)
    print("Number of trainable parameters:")
    model.print_trainable_parameters()

    # --- Gradient checkpointing と input gradients の設定 ---
    print("Configuring gradient checkpointing and input gradients...")

    # 1. enable_input_require_grads() を確実に実行
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
        print("✓ enable_input_require_grads() called on PeftModel")
    else:
        print("⚠ Warning: model does not have enable_input_require_grads()")

    # 2. base_model に対しても明示的に設定（念のため）
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'enable_input_require_grads'):
        model.base_model.enable_input_require_grads()
        print("✓ enable_input_require_grads() called on base_model")

    # 3. gradient checkpointing を backbone に対して有効化
    try:
        # HierPhi4 の場合: model.base_model.backbone
        if hasattr(model.base_model, 'backbone') and hasattr(model.base_model.backbone, 'gradient_checkpointing_enable'):
            model.base_model.backbone.gradient_checkpointing_enable()
            print("✓ gradient_checkpointing_enable() called on backbone")
        # 単純モデルの場合: model.base_model
        elif hasattr(model.base_model, 'gradient_checkpointing_enable'):
            model.base_model.gradient_checkpointing_enable()
            print("✓ gradient_checkpointing_enable() called on base_model")
        # フォールバック
        elif hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("✓ gradient_checkpointing_enable() called on model")
        else:
            print("⚠ Warning: could not find gradient_checkpointing_enable() method")
    except Exception as e:
        print(f"⚠ Warning: failed to enable gradient checkpointing: {e}")

    # 4. requires_grad 状態を確認
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters count: {len(trainable_params)}")
    if len(trainable_params) > 0:
        print(f"Sample trainable parameter requires_grad: {trainable_params[0].requires_grad}")
    else:
        print("⚠ Warning: No trainable parameters found!")

    # device_map="auto"を使用しているため、モデルは既にGPUに配置されています
    # 明示的なcuda()呼び出しは不要
    # （device_map="auto"がbase_modelを自動的に最適配置）
    print(f"Model device: {next(model.parameters()).device}")

    # 追加のメモリ最適化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- トレーニング引数の設定 ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=LOGGING_STEPS,
        metric_for_best_model=None,
        greater_is_better=True,
        # Trainerのベスト判定に依存しない（独自コールバックで保存）
        load_best_model_at_end=False,
        # 予測とラベルを保持してcompute_metricsを必ず実行
        prediction_loss_only=False,
        report_to="wandb" if USE_WANDB else "none",
        bf16=True,  # BF16を使用
        gradient_checkpointing=True,  # メモリ効率化のため有効化
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # メモリ効率向上のため追加
        remove_unused_columns=False,  # カラムを削除しない
        lr_scheduler_type="cosine",  # コサインスケジューラーを使用
        warmup_ratio=0.0,  # ウォームアップを無効化
        save_total_limit=2,
        max_grad_norm=MAX_GRAD_NORM,  # Gradient clipping
        optim="adamw_bnb_8bit" if USE_8BIT_ADAM else "adamw_torch",  # 8-bit Adam optimizer
    )

    # --- トレーナーのセットアップとトレーニング ---
    print("Setting up trainer...")
    # デバッグ: prediction_loss_only を明示的に確認
    print(f"TrainingArguments.prediction_loss_only = {training_args.prediction_loss_only}")

    # エポックあたりのステップ数を計算
    steps_per_epoch = len(train_ds) // (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)  # gradient_accumulation_stepsを考慮
    total_steps = steps_per_epoch * EPOCHS
    print(f"\nDataset statistics:")
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Batch size: {TRAIN_BATCH_SIZE} (with gradient accumulation: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")
    print(f"Evaluation interval: every {EVAL_STEPS} steps (~{EVAL_STEPS/steps_per_epoch:.2f} epochs)")
    print(f"Early stopping after {EARLY_STOPPING_PATIENCE} evaluations without improvement")

    # カスタムデータコレーターを使用
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=MAX_LEN)

    # コールバックの設定
    callbacks = []

    # GradientCheckCallbackを追加（デバッグ用）
    gradient_check_callback = GradientCheckCallback()
    callbacks.append(gradient_check_callback)
    print("GradientCheckCallback enabled - 最初のステップで勾配を確認します")

    # SaveBestMap3Callbackを追加
    save_best_callback = SaveBestMap3Callback(save_dir=OUTPUT_DIR, tokenizer=tokenizer, metric_name=METRIC_NAME)
    callbacks.append(save_best_callback)
    print(f"SaveBestMap3Callback enabled - モデルは {OUTPUT_DIR}/best_map3 に保存されます")

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_map3,
        callbacks=callbacks,
    )

    print("Starting training...")
    # デバッグ: ラベル名
    try:
        print(f"Trainer.label_names = {getattr(trainer, 'label_names', None)}")
    except Exception:
        pass
    trainer.train()

    # --- トレーニング終了後の最終評価 ---
    print("\n" + "="*60)
    print("トレーニング完了 - 最終評価を実行中...")
    print("="*60)
    final_eval_results = trainer.evaluate()
    final_map3 = final_eval_results.get(f"eval_{METRIC_NAME}", 0.0)
    print(f"\n🏁 最終評価結果:")
    print(f"   最終MAP@{MAP_K}スコア: {final_map3:.4f}")
    print(f"   全体のベストMAP@{MAP_K}スコア: {save_best_callback.best_map3:.4f}")

    # 最終評価が新しいベストスコアの場合、明示的に保存
    if final_map3 > save_best_callback.best_map3:
        print(f"🎉 最終評価で新しいベストスコア達成！ {final_map3:.4f} > {save_best_callback.best_map3:.4f}")
        save_best_callback.best_map3 = final_map3
        best_map3_path = os.path.join(OUTPUT_DIR, 'best_map3')
        os.makedirs(best_map3_path, exist_ok=True)
        model.save_pretrained(best_map3_path)
        tokenizer.save_pretrained(best_map3_path)
        print(f"   最終ベストモデルを {best_map3_path} に保存しました")

    # --- モデルの保存 ---
    print("\nSaving model...")
    # LoRAアダプターのみを保存
    model.save_pretrained(BEST_MODEL_PATH)
    # トークナイザーも保存
    tokenizer.save_pretrained(BEST_MODEL_PATH)

    print("Training completed successfully!")
    print(f"Model saved to: {BEST_MODEL_PATH}")
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")

    # WandBの終了
    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
