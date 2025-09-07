"""
共通ユーティリティ関数
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset
import torch


def prepare_correct_answers(train_data):
    """正解答案データを準備"""
    idx = train_data.apply(lambda row: row.Category.split('_')[0] == 'True', axis=1)
    correct = train_data.loc[idx].copy()
    correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
    correct = correct.sort_values('c', ascending=False)
    correct = correct.drop_duplicates(['QuestionId'])[['QuestionId','MC_Answer']]
    correct['is_correct'] = 1
    return correct


def tokenize_dataset(dataset, tokenizer, max_len):
    """データセットをトークナイズ"""
    def tokenize(batch):
        # パディングはDataCollatorで行うため、ここではトークナイズのみ
        return tokenizer(
            batch['text'],
            padding=False,  # パディングはDataCollatorに任せる
            truncation=True,
            max_length=max_len,
            return_tensors=None  # map時は'None'を使用
        )

    dataset = dataset.map(tokenize, batched=True, batch_size=100)
    # columnsの設定時にlabelを保持
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


def filter_data_by_question_id(data, question_id):
    """指定されたQuestionIdでデータをフィルタリング"""
    return data[data['QuestionId'] == question_id].copy()


def get_question_specific_labels(data):
    """QuestionId特有のラベル（targetカラム）を取得"""
    return data['target'].unique()


def save_question_results(question_id, results, save_dir):
    """QuestionId別の結果を保存"""
    import os
    import json
    
    os.makedirs(save_dir, exist_ok=True)
    
    results_file = os.path.join(save_dir, f'question_{question_id}_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results for Question {question_id} saved to: {results_file}")


def create_combined_submission(question_predictions_dict, test_data):
    """QuestionIdごとの予測結果を統合して提出用ファイルを作成"""
    submission_parts = []
    
    for question_id, (predictions, question_test_data, label_encoder) in question_predictions_dict.items():
        # QuestionIdごとの提出用データを作成
        question_submission = create_submission(predictions, question_test_data, label_encoder)
        submission_parts.append(question_submission)
    
    # 全ての結果を統合
    final_submission = pd.concat(submission_parts, ignore_index=True)
    
    # row_idでソートして元の順序を保持
    final_submission = final_submission.sort_values('row_id').reset_index(drop=True)
    
    return final_submission


def print_question_data_summary(data, question_id):
    """QuestionIdごとのデータサマリーを表示"""
    print(f"\n=== Question {question_id} Data Summary ===")
    print(f"Total samples: {len(data)}")
    print(f"Unique labels: {data['target'].nunique()}")
    print(f"Label distribution:")
    print(data['target'].value_counts())
    print("=" * 50)
