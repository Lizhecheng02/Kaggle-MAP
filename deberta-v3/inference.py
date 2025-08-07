"""
推論スクリプト - 高精度推論
"""

import pandas as pd
import numpy as np
import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from datasets import Dataset
import joblib
from tqdm import tqdm

# カスタムモジュールのインポート
from config import *
from utils import prepare_correct_answers, format_input
from multitask_model import MultiTaskDebertaModel


def apply_tta(text, num_augmentations=3):
    """Test Time Augmentation - 推論時のデータ拡張"""
    augmented_texts = [text]  # オリジナルを含む
    
    # 軽微な変更を加えたバージョンを生成
    variations = [
        # 句読点の追加/削除
        lambda t: t.replace(".", ". ").replace("  ", " "),
        # 小文字/大文字の変更（数式は保持）
        lambda t: ' '.join([word.lower() if not any(c.isdigit() for c in word) else word for word in t.split()]),
        # 同義語置換（シンプルなもののみ）
        lambda t: t.replace("because", "since").replace("therefore", "thus"),
    ]
    
    for i in range(min(num_augmentations - 1, len(variations))):
        augmented_texts.append(variations[i](text))
    
    return augmented_texts


def predict_with_tta(model, tokenizer, texts, device):
    """TTA付き予測"""
    all_predictions = []
    
    for text in texts:
        # トークナイズ
        inputs = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors='pt'
        ).to(device)
        
        # 予測
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            all_predictions.append(probs.cpu().numpy())
    
    # 予測を平均化
    avg_predictions = np.mean(all_predictions, axis=0)
    return avg_predictions[0]  # バッチサイズ1なので最初の要素を返す


def main():
    """メイン推論関数"""
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- データの読み込み ---
    print("Loading test data...")
    test = pd.read_csv(TEST_DATA_PATH)
    print(f"Test shape: {test.shape}")
    
    # --- ラベルエンコーダーの読み込み ---
    print("Loading label encoder...")
    le = joblib.load(LABEL_ENCODER_PATH)
    n_classes = len(le.classes_)
    
    # --- 補助エンコーダーの読み込み（マルチタスクの場合）---
    misconception_encoder = None
    if USE_MULTITASK:
        try:
            aux_encoders = joblib.load(AUXILIARY_ENCODERS_PATH)
            misconception_encoder = aux_encoders.get('misconception_encoder')
        except:
            print("Warning: Could not load auxiliary encoders")
    
    # --- 特徴量エンジニアリング ---
    print("Performing feature engineering...")
    # トレーニングデータから正解答案情報を取得
    train_for_correct = pd.read_csv(TRAIN_DATA_PATH)
    correct = prepare_correct_answers(train_for_correct)
    test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    test.is_correct = test.is_correct.fillna(0.5)  # テストデータの不明な答えは0.5
    
    # --- 入力テキストのフォーマット ---
    print("Formatting input text...")
    test['text'] = test.apply(format_input, axis=1)
    
    # --- トークナイザーとモデルの初期化 ---
    print("Loading tokenizer and model...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    
    # モデルのロード
    if USE_MULTITASK:
        num_misconceptions = len(misconception_encoder.classes_) if misconception_encoder else 36
        model = MultiTaskDebertaModel.from_pretrained(
            BEST_MODEL_PATH,
            num_labels=n_classes,
            num_misconceptions=num_misconceptions
        )
    else:
        model = DebertaV2ForSequenceClassification.from_pretrained(
            BEST_MODEL_PATH,
            num_labels=n_classes
        )
    
    model = model.to(device)
    model.eval()
    
    # --- 予測の実行 ---
    print("Making predictions...")
    predictions = []
    
    # バッチ処理の準備
    batch_size = 32
    num_batches = (len(test) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Predicting"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(test))
            batch_texts = test['text'].iloc[start_idx:end_idx].tolist()
            
            batch_predictions = []
            for text in batch_texts:
                if USE_TTA and TTA_ROUNDS > 1:
                    # TTAを使用
                    augmented_texts = apply_tta(text, TTA_ROUNDS)
                    pred = predict_with_tta(model, tokenizer, augmented_texts, device)
                else:
                    # 通常の予測
                    inputs = tokenizer(
                        text,
                        padding='max_length',
                        truncation=True,
                        max_length=MAX_LEN,
                        return_tensors='pt'
                    ).to(device)
                    
                    outputs = model(**inputs)
                    logits = outputs.logits
                    pred = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
                
                batch_predictions.append(pred)
            
            predictions.extend(batch_predictions)
    
    predictions = np.array(predictions)
    
    # --- 予測の後処理 ---
    print("Post-processing predictions...")
    
    # 信頼度に基づく調整
    # 高信頼度の予測を強化
    confidence_threshold = 0.7
    for i in range(len(predictions)):
        max_prob = np.max(predictions[i])
        if max_prob > confidence_threshold:
            # 最大確率をさらに強化
            max_idx = np.argmax(predictions[i])
            predictions[i] = predictions[i] * 0.9
            predictions[i][max_idx] = predictions[i][max_idx] / 0.9 * 1.1
            # 正規化
            predictions[i] = predictions[i] / predictions[i].sum()
    
    # --- 提出ファイルの作成 ---
    print("Creating submission file...")
    
    # Top-3予測の取得
    top3 = np.argsort(-predictions, axis=1)[:, :3]
    flat = top3.flatten()
    decoded = le.inverse_transform(flat)
    top3_labels = decoded.reshape(top3.shape)
    pred_strings = [" ".join(r) for r in top3_labels]
    
    submission = pd.DataFrame({
        'row_id': test.row_id.values,
        'Category:Misconception': pred_strings
    })
    
    # 提出ファイルの保存
    submission.to_csv(SUBMISSION_OUTPUT_PATH, index=False)
    print(f"Submission saved to: {SUBMISSION_OUTPUT_PATH}")
    
    # 予測の統計情報を表示
    print("\nPrediction statistics:")
    print(f"Average top-1 confidence: {np.mean(np.max(predictions, axis=1)):.4f}")
    print(f"Predictions with >70% confidence: {np.sum(np.max(predictions, axis=1) > 0.7)}/{len(predictions)}")
    
    # 最も頻繁に予測されたラベル
    top1_predictions = le.inverse_transform(top3[:, 0])
    pred_counts = pd.Series(top1_predictions).value_counts().head(10)
    print("\nTop 10 most predicted labels:")
    print(pred_counts)


if __name__ == "__main__":
    main()