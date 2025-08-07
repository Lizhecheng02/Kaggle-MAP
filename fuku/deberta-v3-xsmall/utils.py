"""
共通ユーティリティ関数
"""

import pandas as pd
import numpy as np
from transformers import DebertaV2Tokenizer
from datasets import Dataset
import torch
import random
import re
from config import MAX_LEN


def prepare_correct_answers(train_data):
    """正解答案データを準備"""
    idx = train_data.apply(lambda row: row.Category.split('_')[0] == 'True', axis=1)
    correct = train_data.loc[idx].copy()
    correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
    correct = correct.sort_values('c', ascending=False)
    correct = correct.drop_duplicates(['QuestionId'])[['QuestionId','MC_Answer']]
    correct['is_correct'] = 1
    return correct


def format_input(row):
    """入力データをモデル用プロンプトにフォーマット"""
    if row['is_correct']:
        status = "This answer is correct."
    else:
        status = "This answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{status}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )


def tokenize_dataset(dataset, tokenizer, max_len):
    """データセットをトークナイズ"""
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=max_len)
    
    dataset = dataset.map(tokenize, batched=True)
    columns = ['input_ids', 'attention_mask', 'label'] if 'label' in dataset.column_names else ['input_ids', 'attention_mask']
    dataset.set_format(type='torch', columns=columns)
    return dataset


def compute_map3(eval_pred):
    """Top-3 予測に基づくMAP@3を計算"""
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    top3 = np.argsort(-probs, axis=1)[:, :3]
    score = 0.0
    for i, label in enumerate(labels):
        ranks = top3[i]
        if ranks[0] == label:
            score += 1.0
        elif ranks[1] == label:
            score += 1.0 / 2
        elif ranks[2] == label:
            score += 1.0 / 3
    return {"map@3": score / len(labels)}


def create_submission(predictions, test_data, label_encoder):
    """予測結果から提出用ファイルを作成"""
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
    top3 = np.argsort(-probs, axis=1)[:, :3]
    flat = top3.flatten()
    decoded = label_encoder.inverse_transform(flat)
    top3_labels = decoded.reshape(top3.shape)
    pred_strings = [" ".join(r) for r in top3_labels]
    
    submission = pd.DataFrame({
        'row_id': test_data.row_id.values,
        'Category:Misconception': pred_strings
    })
    return submission


def augment_text(text, config):
    """テキストデータ拡張関数"""
    augmented_text = text
    words = text.split()
    
    # 同義語置換
    if random.random() < config.get('synonym_replacement_prob', 0):
        # 簡易版：ランダムな単語の置換（実際の実装では同義語辞書を使用）
        if len(words) > 0:
            idx = random.randint(0, len(words) - 1)
            words[idx] = words[idx] + "_syn"
            augmented_text = " ".join(words)
    
    # ランダム挿入
    if random.random() < config.get('random_insertion_prob', 0):
        if len(words) > 0:
            idx = random.randint(0, len(words))
            words.insert(idx, "RANDOM_WORD")
            augmented_text = " ".join(words)
    
    # ランダムスワップ
    if random.random() < config.get('random_swap_prob', 0):
        if len(words) > 1:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            augmented_text = " ".join(words)
    
    # ランダム削除
    if random.random() < config.get('random_deletion_prob', 0):
        if len(words) > 1:
            words = [w for w in words if random.random() > 0.1]
            augmented_text = " ".join(words)
    
    return augmented_text


def augment_dataset(dataset, tokenizer, augmentation_config):
    """データセット全体にデータ拡張を適用"""
    def augment_batch(batch):
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(batch['text'], batch['label']):
            # オリジナルデータを追加
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            # 拡張データを追加
            augmented_text = augment_text(text, augmentation_config)
            augmented_texts.append(augmented_text)
            augmented_labels.append(label)
        
        # トークナイズ
        tokenized = tokenizer(
            augmented_texts, 
            padding='max_length', 
            truncation=True, 
            max_length=MAX_LEN
        )
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'label': augmented_labels
        }
    
    # バッチサイズを小さくして処理
    augmented_dataset = dataset.map(
        augment_batch,
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names
    )
    
    augmented_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return augmented_dataset