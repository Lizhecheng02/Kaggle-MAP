"""
トークナイザーの語彙サイズとトークンIDの範囲を検証するスクリプト
"""
import pandas as pd
from transformers import AutoTokenizer
from config import MODEL_NAME, TRAIN_DATA_PATH, MAX_LEN
from utils import prepare_correct_answers, format_input

# トークナイザーを読み込む
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# トークナイザーの情報を表示
print(f"\n=== Tokenizer Information ===")
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Model max length: {tokenizer.model_max_length}")
print(f"Pad token: {tokenizer.pad_token}")
print(f"Pad token ID: {tokenizer.pad_token_id}")
print(f"EOS token: {tokenizer.eos_token}")
print(f"EOS token ID: {tokenizer.eos_token_id}")

# パディングトークンの設定
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"\nSet pad_token to eos_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

# サンプルデータを読み込んでトークナイズ
print("\n=== Testing tokenization on sample data ===")
train = pd.read_csv(TRAIN_DATA_PATH)
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category + ":" + train.Misconception

# 最初の数サンプルで確認
correct = prepare_correct_answers(train)
train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)
train['text'] = train.apply(format_input, axis=1)

# トークナイズテスト
sample_texts = train['text'].head(5).tolist()
for i, text in enumerate(sample_texts):
    print(f"\n--- Sample {i+1} ---")
    print(f"Text length: {len(text)} characters")
    
    # トークナイズ
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
    
    # トークンIDの統計情報
    input_ids = tokens['input_ids'].squeeze().tolist()
    non_pad_ids = [tid for tid in input_ids if tid != tokenizer.pad_token_id]
    
    print(f"Number of tokens (non-padding): {len(non_pad_ids)}")
    print(f"Max token ID: {max(non_pad_ids)}")
    print(f"Min token ID: {min(non_pad_ids)}")
    
    # 語彙サイズを超えるトークンIDがあるかチェック
    over_vocab = [tid for tid in non_pad_ids if tid >= tokenizer.vocab_size]
    if over_vocab:
        print(f"WARNING: Found {len(over_vocab)} token IDs over vocabulary size!")
        print(f"Over-vocabulary token IDs: {over_vocab[:10]}...")  # 最初の10個を表示

# バッチでのトークナイズテスト
print("\n=== Batch tokenization test ===")
batch_texts = train['text'].head(32).tolist()  # バッチサイズと同じ数
batch_tokens = tokenizer(batch_texts, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')

print(f"Batch shape: {batch_tokens['input_ids'].shape}")
all_ids = batch_tokens['input_ids'].flatten().tolist()
non_pad_ids = [tid for tid in all_ids if tid != tokenizer.pad_token_id]
print(f"Max token ID in batch: {max(non_pad_ids)}")
print(f"Min token ID in batch: {min(non_pad_ids)}")

# 統計情報
over_vocab_count = sum(1 for tid in non_pad_ids if tid >= tokenizer.vocab_size)
if over_vocab_count > 0:
    print(f"\nWARNING: {over_vocab_count} tokens exceed vocabulary size!")
    print(f"This will cause indexing errors during training.")