"""
MathBERT動作確認スクリプト
"""

import sys
sys.path.append('/root/kaggle/myenv/lib/python3.8/site-packages')

from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 設定ファイルから読み込み
from config import MODEL_NAME

def test_mathbert_loading():
    """MathBERTモデルの読み込みテスト"""
    print(f"Testing MathBERT loading from: {MODEL_NAME}")
    
    try:
        # トークナイザーの読み込み
        print("Loading tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        print("✓ Tokenizer loaded successfully")
        
        # モデルの読み込み（分類タスク用）
        print("Loading model for sequence classification...")
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=100)
        print("✓ Model loaded successfully")
        
        # 簡単なテスト
        test_text = "If a student solves 2x + 3 = 7, what is the value of x?"
        print(f"\nTest text: {test_text}")
        
        # トークナイズ
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        print(f"Input shape: {inputs['input_ids'].shape}")
        
        # 推論（評価モード）
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            print(f"Output shape: {logits.shape}")
            
        print("\n✅ All tests passed! MathBERT is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {type(e).__name__}: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    test_mathbert_loading()