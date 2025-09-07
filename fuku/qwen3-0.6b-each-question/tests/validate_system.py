#!/usr/bin/env python3
"""
QuestionId別システムの簡単なバリデーションスクリプト

学習やテストを実行する前に、システムが正常に設定されているかを確認します。
"""

import os
import sys
import pandas as pd

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def validate_config():
    """設定ファイルの検証"""
    print("🔍 設定ファイルを検証中...")
    
    try:
        from config import (
            QUESTION_IDS, MODEL_NAME, TRAIN_DATA_PATH, TEST_DATA_PATH,
            get_question_output_dir, get_question_model_path, get_question_label_encoder_path
        )
        print(f"✅ 設定ファイルが正常に読み込まれました")
        print(f"  - QuestionID数: {len(QUESTION_IDS)}")
        print(f"  - モデル名: {MODEL_NAME}")
        return True
    except ImportError as e:
        print(f"❌ 設定ファイルの読み込みに失敗: {e}")
        return False

def validate_data_files():
    """データファイルの存在確認"""
    print("\n🔍 データファイルを検証中...")
    
    try:
        from config import TRAIN_DATA_PATH, TEST_DATA_PATH
        
        # 学習データの確認
        if os.path.exists(TRAIN_DATA_PATH):
            train_df = pd.read_csv(TRAIN_DATA_PATH)
            print(f"✅ 学習データが見つかりました: {TRAIN_DATA_PATH}")
            print(f"  - 形状: {train_df.shape}")
            print(f"  - QuestionId数: {train_df['QuestionId'].nunique()}")
        else:
            print(f"⚠️  学習データが見つかりません: {TRAIN_DATA_PATH}")
        
        # テストデータの確認
        if os.path.exists(TEST_DATA_PATH):
            test_df = pd.read_csv(TEST_DATA_PATH)
            print(f"✅ テストデータが見つかりました: {TEST_DATA_PATH}")
            print(f"  - 形状: {test_df.shape}")
            print(f"  - QuestionId数: {test_df['QuestionId'].nunique()}")
        else:
            print(f"⚠️  テストデータが見つかりません: {TEST_DATA_PATH}")
        
        return True
    except Exception as e:
        print(f"❌ データファイルの検証に失敗: {e}")
        return False

def validate_utilities():
    """ユーティリティ関数の検証"""
    print("\n🔍 ユーティリティ関数を検証中...")
    
    try:
        from utils import (
            filter_data_by_question_id, get_question_specific_labels,
            save_question_results, create_combined_submission
        )
        print("✅ ユーティリティ関数が正常に読み込まれました")
        return True
    except ImportError as e:
        print(f"❌ ユーティリティ関数の読み込みに失敗: {e}")
        return False

def validate_prompts():
    """プロンプト関数の検証"""
    print("\n🔍 プロンプト関数を検証中...")
    
    try:
        from prompts import prompt_registry
        from config import PROMPT_VERSION
        
        if PROMPT_VERSION in prompt_registry:
            print(f"✅ プロンプト関数が見つかりました: {PROMPT_VERSION}")
        else:
            print(f"❌ 指定されたプロンプト関数が見つかりません: {PROMPT_VERSION}")
            print(f"利用可能な関数: {list(prompt_registry.keys())}")
            return False
        
        return True
    except ImportError as e:
        print(f"❌ プロンプト関数の読み込みに失敗: {e}")
        return False

def validate_question_data_distribution():
    """QuestionIdごとのデータ分布確認"""
    print("\n🔍 QuestionIdごとのデータ分布を確認中...")
    
    try:
        from config import TRAIN_DATA_PATH, QUESTION_IDS
        from utils import filter_data_by_question_id
        
        if not os.path.exists(TRAIN_DATA_PATH):
            print(f"⚠️  学習データが見つからないため、分布確認をスキップします")
            return True
        
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        train_df.Misconception = train_df.Misconception.fillna('NA')
        train_df['target'] = train_df.Category + ":" + train_df.Misconception
        
        print(f"\nQuestionIdごとのデータ分布:")
        print("-" * 50)
        
        total_samples = 0
        questions_with_data = 0
        
        for question_id in QUESTION_IDS:
            question_data = filter_data_by_question_id(train_df, question_id)
            n_samples = len(question_data)
            n_labels = question_data['target'].nunique() if n_samples > 0 else 0
            
            status = "✅" if n_samples > 0 else "❌"
            print(f"{status} Question {question_id}: {n_samples:4d} samples, {n_labels:2d} labels")
            
            if n_samples > 0:
                questions_with_data += 1
                total_samples += n_samples
        
        print("-" * 50)
        print(f"合計: {questions_with_data}/{len(QUESTION_IDS)} 問題にデータがあります")
        print(f"総サンプル数: {total_samples}")
        
        return True
        
    except Exception as e:
        print(f"❌ データ分布の確認に失敗: {e}")
        return False

def validate_dependencies():
    """依存関係の確認"""
    print("\n🔍 依存関係を確認中...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'transformers', 
        'torch', 'datasets', 'joblib'
    ]
    
    optional_packages = [
        'peft', 'wandb'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (必須)")
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package} (オプション)")
        except ImportError:
            print(f"⚠️  {package} (オプション)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n❌ 必須パッケージが不足しています: {missing_required}")
        return False
    
    if missing_optional:
        print(f"\n⚠️  オプションパッケージが不足していますが、動作には影響ありません: {missing_optional}")
    
    return True

def main():
    """メイン検証関数"""
    print("="*80)
    print("🔍 QuestionId別システムバリデーション")
    print("="*80)
    
    all_checks_passed = True
    
    # 各種検証を実行
    checks = [
        ("設定ファイル", validate_config),
        ("依存関係", validate_dependencies),
        ("データファイル", validate_data_files),
        ("ユーティリティ関数", validate_utilities),
        ("プロンプト関数", validate_prompts),
        ("データ分布", validate_question_data_distribution),
    ]
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_checks_passed = False
        except Exception as e:
            print(f"\n❌ {check_name}の検証中にエラーが発生: {e}")
            all_checks_passed = False
    
    # 最終結果
    print("\n" + "="*80)
    if all_checks_passed:
        print("🎉 すべての検証が成功しました!")
        print("システムは学習・推論の準備ができています。")
        print("\n次のステップ:")
        print("1. 学習を開始: python train.py")
        print("2. 推論を実行: python submit.py")
    else:
        print("❌ 一部の検証が失敗しました。")
        print("問題を修正してから学習・推論を実行してください。")
    print("="*80)
    
    return all_checks_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)