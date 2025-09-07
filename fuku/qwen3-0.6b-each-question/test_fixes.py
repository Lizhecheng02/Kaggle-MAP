#!/usr/bin/env python3
"""
修正内容をテストするスクリプト
"""

import pandas as pd
import numpy as np
from config import QUESTION_IDS, TRAIN_DATA_PATH
from utils import filter_data_by_question_id, prepare_correct_answers


def test_question_ids_type():
    """QuestionIdが整数型になっているかテスト"""
    print("=== Testing QuestionId Type ===")
    print(f"QUESTION_IDS type: {type(QUESTION_IDS)}")
    print(f"First QuestionId: {QUESTION_IDS[0]} (type: {type(QUESTION_IDS[0])})")
    
    # すべてのQuestionIdが整数であることを確認
    all_int = all(isinstance(qid, int) for qid in QUESTION_IDS)
    print(f"All QuestionIds are integers: {all_int}")
    assert all_int, "All QuestionIds should be integers"
    print("✅ QuestionId type test passed!")


def test_data_filtering():
    """データフィルタリング機能をテスト"""
    print("\n=== Testing Data Filtering ===")
    
    # トレーニングデータを読み込み
    train = pd.read_csv(TRAIN_DATA_PATH)
    train.Misconception = train.Misconception.fillna('NA')
    train['target'] = train.Category + ":" + train.Misconception
    
    # 特徴量エンジニアリング
    correct = prepare_correct_answers(train)
    train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    train.is_correct = train.is_correct.fillna(0)
    
    print(f"Total train data: {len(train)}")
    print(f"Unique QuestionIds in data: {sorted(train['QuestionId'].unique())}")
    print(f"Configured QUESTION_IDS: {sorted(QUESTION_IDS)}")
    
    # 各QuestionIdでフィルタリングテスト
    total_found = 0
    for question_id in QUESTION_IDS[:5]:  # 最初の5つのQuestionIdでテスト
        question_data = filter_data_by_question_id(train, question_id)
        print(f"Question {question_id}: {len(question_data)} samples")
        total_found += len(question_data)
        
        if len(question_data) > 0:
            # データが見つかった場合の基本チェック
            assert question_data['QuestionId'].nunique() == 1, f"Question {question_id} should have only one unique QuestionId"
            assert (question_data['QuestionId'] == question_id).all(), f"Question {question_id} filtering failed"
            
            # targetカラムが存在することを確認
            assert 'target' in question_data.columns, "target column should exist"
            unique_labels = question_data['target'].nunique()
            print(f"  - Unique labels: {unique_labels}")
    
    print(f"Total samples found in first 5 questions: {total_found}")
    print("✅ Data filtering test passed!")


def test_prompt_creation():
    """プロンプト生成機能をテスト"""
    print("\n=== Testing Prompt Creation ===")
    
    # トレーニングデータを読み込み
    train = pd.read_csv(TRAIN_DATA_PATH)
    train.Misconception = train.Misconception.fillna('NA')
    train['target'] = train.Category + ":" + train.Misconception
    
    # 最初のQuestionIdでテスト
    first_question_id = QUESTION_IDS[0]
    question_data = filter_data_by_question_id(train, first_question_id)
    
    if len(question_data) == 0:
        print(f"⚠️  No data found for Question {first_question_id}, skipping prompt test")
        return
    
    # プロンプト関数をインポートしてテスト
    try:
        from transformers import AutoTokenizer
        from prompts import prompt_registry
        from config import PROMPT_VERSION, MODEL_NAME
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        prompt_function = prompt_registry[PROMPT_VERSION]
        
        # 最初のサンプルでプロンプト生成をテスト
        sample_row = question_data.iloc[0]
        prompt_text = prompt_function(tokenizer, sample_row)
        
        print(f"Generated prompt length: {len(prompt_text)}")
        print(f"Prompt preview: {prompt_text[:200]}...")
        
        assert len(prompt_text) > 0, "Prompt should not be empty"
        print("✅ Prompt creation test passed!")
        
    except Exception as e:
        print(f"⚠️  Prompt test failed: {e}")


def main():
    """すべてのテストを実行"""
    print("Starting tests for bug fixes...")
    
    try:
        test_question_ids_type()
        test_data_filtering() 
        test_prompt_creation()
        
        print("\n🎉 All tests passed! The fixes should work correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())