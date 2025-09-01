"""
CoT統合のテストスクリプト
"""

import pandas as pd
import os
from utils import format_input, prepare_correct_answers
from config import TRAIN_DATA_PATH

def test_cot_integration():
    """CoT統合が正しく動作するかテスト"""
    
    print("=" * 80)
    print("CoT Integration Test")
    print("=" * 80)
    
    # トレーニングデータの読み込み
    print("\n1. Loading training data...")
    train = pd.read_csv(TRAIN_DATA_PATH)
    train.Misconception = train.Misconception.fillna('NA')
    train['target'] = train.Category + ":" + train.Misconception
    print(f"   Train shape: {train.shape}")
    
    # CoTデータの読み込み
    print("\n2. Loading CoT data...")
    cot_data_path = 'train_cot_gemini_2_5_flash.parquet'
    if os.path.exists(cot_data_path):
        cot_data = pd.read_parquet(cot_data_path)
        print(f"   CoT data shape: {cot_data.shape}")
        
        # マージ
        train = train.merge(cot_data[['row_id', 'think']], on='row_id', how='left')
        print(f"   Merged successfully. CoT available for {train.think.notna().sum()}/{len(train)} samples")
    else:
        print(f"   ERROR: CoT data file not found!")
        return
    
    # 特徴量エンジニアリング
    print("\n3. Performing feature engineering...")
    correct = prepare_correct_answers(train)
    train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    train.is_correct = train.is_correct.fillna(0)
    
    # CoTありとなしの例を取得
    print("\n4. Testing format_input function...")
    
    # CoTありの例
    cot_sample = train[train.think.notna()].iloc[0]
    print("\n   Sample WITH CoT:")
    print(f"   - row_id: {cot_sample['row_id']}")
    print(f"   - Question: {cot_sample['QuestionText'][:100]}...")
    print(f"   - CoT think (first 200 chars): {cot_sample['think'][:200]}...")
    
    formatted_with_cot = format_input(cot_sample)
    print("\n   Formatted prompt WITH CoT:")
    print("-" * 40)
    print(formatted_with_cot[:500])  # 最初の500文字を表示
    print("-" * 40)
    
    # CoTなしの例
    no_cot_sample = train[train.think.isna()].iloc[0] if train.think.isna().any() else None
    if no_cot_sample is not None:
        print("\n   Sample WITHOUT CoT:")
        print(f"   - row_id: {no_cot_sample['row_id']}")
        print(f"   - Question: {no_cot_sample['QuestionText'][:100]}...")
        
        formatted_without_cot = format_input(no_cot_sample)
        print("\n   Formatted prompt WITHOUT CoT:")
        print("-" * 40)
        print(formatted_without_cot[:500])  # 最初の500文字を表示
        print("-" * 40)
    
    # 統計情報
    print("\n5. Statistics:")
    print(f"   - Total samples: {len(train)}")
    print(f"   - Samples with CoT: {train.think.notna().sum()} ({train.think.notna().sum()/len(train)*100:.1f}%)")
    print(f"   - Samples without CoT: {train.think.isna().sum()} ({train.think.isna().sum()/len(train)*100:.1f}%)")
    
    # CoT内容の分析
    print("\n6. CoT content analysis:")
    cot_lengths = train[train.think.notna()]['think'].str.len()
    print(f"   - Average CoT length: {cot_lengths.mean():.0f} characters")
    print(f"   - Min CoT length: {cot_lengths.min():.0f} characters")
    print(f"   - Max CoT length: {cot_lengths.max():.0f} characters")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    test_cot_integration()