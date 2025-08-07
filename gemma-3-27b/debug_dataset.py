"""
データセットのトークンIDとパディング処理をデバッグするスクリプト
"""
import pandas as pd
import torch
from transformers import AutoTokenizer
from datasets import Dataset
from config import MODEL_NAME, TRAIN_DATA_PATH, MAX_LEN
from utils import prepare_correct_answers, format_input

# トークナイザーを読み込む
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# パディングトークンの設定
if tokenizer.pad_token is None:
    tokenizer.pad_token_id = 0
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)

print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Pad token ID: {tokenizer.pad_token_id}")

# データの準備
train = pd.read_csv(TRAIN_DATA_PATH)
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category + ":" + train.Misconception

# LabelEncoderを使用してラベルを作成
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['label'] = le.fit_transform(train['target'])

# サンプルデータの準備
correct = prepare_correct_answers(train)
train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)
train['text'] = train.apply(format_input, axis=1)

# 少数のサンプルでテスト
sample_texts = train['text'].head(3).tolist()
labels = train['label'].head(3).tolist()

print("\n=== Manual tokenization test ===")
for i, text in enumerate(sample_texts):
    print(f"\nSample {i+1}:")
    
    # 手動でトークナイズ（paddingなし）
    tokens_no_pad = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors='pt')
    print(f"Without padding - shape: {tokens_no_pad['input_ids'].shape}")
    print(f"Token IDs (first 20): {tokens_no_pad['input_ids'][0][:20].tolist()}")
    print(f"Max token ID: {tokens_no_pad['input_ids'].max().item()}")
    
    # longest paddingでトークナイズ
    tokens_longest = tokenizer([text], padding='longest', truncation=True, max_length=MAX_LEN, return_tensors='pt')
    print(f"With 'longest' padding - shape: {tokens_longest['input_ids'].shape}")
    
    # max_length paddingでトークナイズ
    tokens_max = tokenizer([text], padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
    print(f"With 'max_length' padding - shape: {tokens_max['input_ids'].shape}")
    print(f"Last 10 token IDs: {tokens_max['input_ids'][0][-10:].tolist()}")

# バッチ処理のテスト
print("\n=== Batch processing test ===")
batch_texts = sample_texts

# padding='longest'でバッチ処理
batch_tokens = tokenizer(batch_texts, padding='longest', truncation=True, max_length=MAX_LEN, return_tensors='pt')
print(f"Batch shape with 'longest': {batch_tokens['input_ids'].shape}")
print(f"Max token ID in batch: {batch_tokens['input_ids'].max().item()}")
print(f"Min token ID in batch: {batch_tokens['input_ids'].min().item()}")

# input_idsの詳細な検査
unique_ids = torch.unique(batch_tokens['input_ids'])
print(f"Number of unique token IDs: {len(unique_ids)}")
if (unique_ids >= tokenizer.vocab_size).any():
    over_vocab = unique_ids[unique_ids >= tokenizer.vocab_size]
    print(f"WARNING: Token IDs exceeding vocab size: {over_vocab.tolist()}")

# データセット形式でのテスト
print("\n=== Dataset format test ===")
test_df = pd.DataFrame({'text': sample_texts, 'label': labels})
test_ds = Dataset.from_pandas(test_df)

def tokenize_func(batch):
    return tokenizer(
        batch['text'], 
        padding='longest',
        truncation=True, 
        max_length=MAX_LEN,
        return_tensors=None
    )

# トークナイズしてデータセットを作成
test_ds = test_ds.map(tokenize_func, batched=True, batch_size=2)
test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

print(f"Dataset features: {test_ds.features}")
print(f"First sample input_ids shape: {test_ds[0]['input_ids'].shape}")
print(f"First sample max token ID: {test_ds[0]['input_ids'].max().item()}")