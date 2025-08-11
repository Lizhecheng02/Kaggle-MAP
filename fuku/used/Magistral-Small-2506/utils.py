"""
共通ユーティリティ関数
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset
import torch
import os
from mistral_common.tokens.tokenizers.tekken import Tekkenizer


class MistralTokenizerWrapper:
    """Mistral Tekken tokenizer wrapper for Transformers compatibility"""
    def __init__(self, tekken_path, model_config=None):
        self.tokenizer = Tekkenizer.from_file(tekken_path)
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.bos_token = "<s>"
        self.bos_token_id = 1
        self.model_config = model_config
        
    def __call__(self, texts, padding=False, truncation=True, max_length=None, return_tensors=None):
        """Tokenize texts in a transformers-compatible way"""
        if isinstance(texts, str):
            texts = [texts]
        
        all_input_ids = []
        all_attention_mask = []
        
        for text in texts:
            # Encode text with BOS/EOS tokens
            tokens = self.tokenizer.encode(text, bos=True, eos=True)
            
            # Truncate if needed
            if truncation and max_length and len(tokens) > max_length:
                tokens = tokens[:max_length-1] + [self.eos_token_id]
            
            all_input_ids.append(tokens)
            all_attention_mask.append([1] * len(tokens))
        
        # Padding
        if padding and len(texts) > 1:
            max_len = max(len(ids) for ids in all_input_ids)
            for i in range(len(all_input_ids)):
                pad_len = max_len - len(all_input_ids[i])
                all_input_ids[i] = all_input_ids[i] + [self.pad_token_id] * pad_len
                all_attention_mask[i] = all_attention_mask[i] + [0] * pad_len
        
        # Return format
        result = {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask
        }
        
        if return_tensors == 'pt':
            result['input_ids'] = torch.tensor(result['input_ids'])
            result['attention_mask'] = torch.tensor(result['attention_mask'])
        
        return result
    
    def encode(self, text, **kwargs):
        """Encode text to token IDs"""
        return self.tokenizer.encode(text, bos=True, eos=True)
    
    def decode(self, token_ids, **kwargs):
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids)
    
    def save_pretrained(self, save_directory):
        """Save tokenizer (placeholder for compatibility)"""
        os.makedirs(save_directory, exist_ok=True)
        # In a real implementation, we'd save the tekken.json file here


def load_mistral_tokenizer(model_path):
    """Load Mistral tokenizer from tekken.json"""
    tekken_path = os.path.join(model_path, "tekken.json")
    if not os.path.exists(tekken_path):
        raise FileNotFoundError(f"tekken.json not found in {model_path}")
    return MistralTokenizerWrapper(tekken_path)


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
    if row["is_correct"]:
        status = "Yes"
    else:
        status = "No"

    # Magistral-Small-2506用のプロンプトフォーマット
    prompt = (
        "[INST] "
        f"[Mathematical Misconception Analysis Task]\n\n"
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct?: {status}\n"
        f"Explanation: {row['StudentExplanation']}\n"
        "[/INST]\n"
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
