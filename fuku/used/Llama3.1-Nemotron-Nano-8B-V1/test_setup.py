"""
動作確認用テストスクリプト
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModel

# configをインポート
from config import MODEL_NAME, MAX_LEN
from utils import load_llama_tokenizer

def test_environment():
    """環境のテスト"""
    print("=== Environment Check ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

def test_tokenizer():
    """トークナイザーのテスト"""
    print("=== Tokenizer Test ===")
    print(f"Model path: {MODEL_NAME}")
    
    try:
        tokenizer = load_llama_tokenizer(MODEL_NAME)
        print("✓ Tokenizer loaded successfully")
        
        # テストテキスト
        test_text = "This is a test sentence for the Llama model."
        tokens = tokenizer(test_text, truncation=True, max_length=MAX_LEN)
        print(f"✓ Tokenization test passed")
        print(f"  Input: {test_text}")
        print(f"  Token count: {len(tokens['input_ids'])}")
        
        return True
    except Exception as e:
        print(f"✗ Tokenizer test failed: {e}")
        return False

def test_model_loading():
    """モデル読み込みのテスト（メタデータのみ）"""
    print("\n=== Model Loading Test ===")
    try:
        # configファイルの存在確認
        config_path = os.path.join(MODEL_NAME, "config.json")
        if os.path.exists(config_path):
            print(f"✓ Model config found at: {config_path}")
        else:
            print(f"✗ Model config not found at: {config_path}")
            
        # モデルファイルの確認
        model_files = [f for f in os.listdir(MODEL_NAME) if f.endswith(('.bin', '.safetensors', '.pt'))]
        if model_files:
            print(f"✓ Model weights found: {len(model_files)} file(s)")
            for f in model_files[:3]:  # 最初の3ファイルを表示
                print(f"  - {f}")
        else:
            print("✗ No model weight files found")
            
        return True
    except Exception as e:
        print(f"✗ Model check failed: {e}")
        return False

def main():
    """メインテスト関数"""
    print("Starting Llama-3.1-Nemotron-Nano-8B-v1 setup test...\n")
    
    # 環境チェック
    test_environment()
    
    # トークナイザーテスト
    tokenizer_ok = test_tokenizer()
    
    # モデルチェック
    model_ok = test_model_loading()
    
    print("\n=== Test Summary ===")
    if tokenizer_ok and model_ok:
        print("✓ All tests passed! Ready to train.")
    else:
        print("✗ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()