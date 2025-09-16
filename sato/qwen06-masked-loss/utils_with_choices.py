"""
共通ユーティリティ関数 - 選択肢付きバージョン
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


def prepare_answer_choices(train_data, mapping_file='/home/sato/project/map/question_answer_choice_mapping.csv'):
    """各問題のMC_Answer選択肢を準備（マッピングファイルを使用、小文字ラベル）"""
    # マッピングファイルを読み込み
    mapping_df = pd.read_csv(mapping_file)
    
    # 各QuestionIdごとに選択肢を作成
    choices_list = []
    
    for question_id in train_data['QuestionId'].unique():
        # 該当QuestionIdのマッピングを取得
        question_mapping = mapping_df[mapping_df['QuestionId'] == question_id].copy()
        
        if len(question_mapping) > 0:
            # Choice（A,B,C,D）でソート
            question_mapping = question_mapping.sort_values('Choice')
            # 選択肢文字列を作成（小文字ラベル）
            choice_items = []
            choice_mapping = {}  # MC_Answer -> choice label のマッピング
            for _, row in question_mapping.iterrows():
                lowercase_choice = row['Choice'].lower()  # A -> a, B -> b, etc.
                choice_items.append(f"{lowercase_choice}. {row['MC_Answer']}")
                choice_mapping[row['MC_Answer']] = lowercase_choice
            answer_choices_str = '\n'.join(choice_items)
        else:
            # マッピングがない場合は従来の番号方式にフォールバック
            question_answers = train_data[train_data['QuestionId'] == question_id]['MC_Answer'].unique()
            choice_items = []
            choice_mapping = {}
            for i, ans in enumerate(question_answers):
                lowercase_choice = chr(ord('a') + i)  # a, b, c, d, ...
                choice_items.append(f"{lowercase_choice}. {ans}")
                choice_mapping[ans] = lowercase_choice
            answer_choices_str = '\n'.join(choice_items)
        
        choices_list.append({
            'QuestionId': question_id,
            'answer_choices_str': answer_choices_str,
            'choice_mapping': choice_mapping  # MC_Answer -> choice label のマッピングも保存
        })
    
    choices = pd.DataFrame(choices_list)
    return choices


def format_input(row):
    """入力データをモデル用プロンプトにフォーマット（選択肢付き、回答をラベルに変換）"""
    if row["is_correct"]:
        status = "Yes"
    else:
        status = "No"

    # MC_Answerを選択肢ラベル（a, b, c, d）に変換
    student_answer_label = row.get('choice_label', row['MC_Answer'])  # フォールバック

    # Qwen2.5-Math用の数学タスクに特化したプロンプト（選択肢付き）
    prompt = (
        "<|im_start|>user"
        f"[Mathematical Misconception Analysis Task]\n\n"
        f"Question: {row['QuestionText']}\n\n"
        f"Available Answer Choices:\n{row['answer_choices_str']}\n\n"
        f"Student's Answer: {student_answer_label}\n"
        f"Correct?: {status}\n"
        f"Student's Explanation: {row['StudentExplanation']}\n\n"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    return prompt


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
    print(f"[DEBUG] *** compute_map3 function called! ***")
    print(f"[DEBUG] eval_pred type: {type(eval_pred)}")
    print(f"[DEBUG] eval_pred: {eval_pred}")
    
    try:
        logits, labels = eval_pred
        print(f"[DEBUG] compute_map3 called with logits shape: {logits.shape}, labels shape: {labels.shape}")
        
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
        
        map3_score = score / len(labels)
        result = {"eval_map@3": map3_score}
        print(f"[DEBUG] compute_map3 returning: {result}")
        print(f"[DEBUG] *** compute_map3 function completed successfully! ***")
        return result
        
    except Exception as e:
        print(f"[ERROR] compute_map3 failed: {e}")
        print(f"[ERROR] Exception type: {type(e)}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        return {"eval_map@3": 0.0}


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