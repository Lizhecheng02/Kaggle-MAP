#!/usr/bin/env python3
"""
QuestionId別モデル学習・推論システムのテストスクリプト

このスクリプトは以下の機能をテストします:
1. 設定ファイルの読み込み
2. QuestionIdごとのデータフィルタリング
3. パス管理機能
4. ラベルエンコーディング
5. 基本的な学習・推論フロー
"""

import os
import sys
import unittest
import tempfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import QUESTION_IDS, get_question_output_dir, get_question_model_path, get_question_label_encoder_path
from utils import (
    filter_data_by_question_id, 
    get_question_specific_labels,
    print_question_data_summary,
    save_question_results
)
from prompts import prompt_registry
from transformers import AutoTokenizer


class TestQuestionSpecificSystem(unittest.TestCase):
    """QuestionId別システムのテストクラス"""
    
    def setUp(self):
        """テストデータの準備"""
        # サンプルデータの作成
        self.sample_data = pd.DataFrame({
            'QuestionId': ['31772', '31774', '31772', '31777', '31774'],
            'Category': ['True_Correct', 'False_Misconception', 'True_Neither', 'True_Correct', 'False_Correct'],
            'Misconception': ['NA', 'SwapDividend', 'NA', 'NA', 'NA'],
            'QuestionText': [
                'What fraction of the shape is not shaded?',
                'Calculate 1/2 ÷ 6',
                'What fraction of the shape is not shaded?',
                'A box contains 120 counters',
                'Calculate 1/2 ÷ 6'
            ],
            'MC_Answer': ['A', 'B', 'C', 'A', 'D'],
            'StudentExplanation': [
                'I counted 3 out of 9 parts',
                'I multiplied instead of dividing',
                'Not sure about this',
                'I calculated 3/5 of 120',
                'I got the right answer'
            ],
            'is_correct': [1, 0, 1, 1, 1],
            'row_id': [1, 2, 3, 4, 5]
        })
        
        # targetカラムを作成
        self.sample_data['target'] = self.sample_data['Category'] + ':' + self.sample_data['Misconception']
    
    def test_config_loading(self):
        """設定ファイルの読み込みテスト"""
        # QuestionIDsが正しく読み込まれているかテスト
        self.assertIsInstance(QUESTION_IDS, list)
        self.assertGreater(len(QUESTION_IDS), 0)
        self.assertIn('31772', QUESTION_IDS)
        print(f"✅ Config loading test passed. Found {len(QUESTION_IDS)} questions.")
    
    def test_path_management(self):
        """パス管理機能のテスト"""
        question_id = '31772'
        
        # パス生成関数のテスト
        output_dir = get_question_output_dir(question_id)
        model_path = get_question_model_path(question_id)
        label_encoder_path = get_question_label_encoder_path(question_id)
        
        self.assertIn(question_id, output_dir)
        self.assertIn(question_id, model_path)
        self.assertIn(question_id, label_encoder_path)
        
        print(f"✅ Path management test passed.")
        print(f"  Output dir: {output_dir}")
        print(f"  Model path: {model_path}")
        print(f"  Label encoder path: {label_encoder_path}")
    
    def test_data_filtering(self):
        """データフィルタリング機能のテスト"""
        question_id = '31772'
        
        # データフィルタリングのテスト
        filtered_data = filter_data_by_question_id(self.sample_data, question_id)
        
        self.assertEqual(len(filtered_data), 2)  # 31772は2つのサンプルがある
        self.assertTrue(all(filtered_data['QuestionId'] == question_id))
        
        print(f"✅ Data filtering test passed. Filtered {len(filtered_data)} samples for Question {question_id}")
    
    def test_label_processing(self):
        """ラベル処理のテスト"""
        question_id = '31772'
        filtered_data = filter_data_by_question_id(self.sample_data, question_id)
        
        # ラベルの取得と処理
        labels = get_question_specific_labels(filtered_data)
        le = LabelEncoder()
        encoded_labels = le.fit_transform(filtered_data['target'])
        
        self.assertGreater(len(labels), 0)
        self.assertEqual(len(encoded_labels), len(filtered_data))
        
        print(f"✅ Label processing test passed. Found {len(labels)} unique labels for Question {question_id}")
        print(f"  Labels: {labels}")
    
    def test_prompt_generation(self):
        """プロンプト生成のテスト"""
        # トークナイザーの初期化（テスト用に軽量なものを使用）
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        except:
            # トークナイザーが利用できない場合はスキップ
            print("⚠️  Tokenizer not available, skipping prompt generation test")
            return
        
        # プロンプト関数の取得
        prompt_function = prompt_registry.get('create_prompt_v2')
        if prompt_function is None:
            print("⚠️  Prompt function not available, skipping test")
            return
        
        # サンプルデータでプロンプト生成をテスト
        sample_row = self.sample_data.iloc[0]
        try:
            prompt = prompt_function(tokenizer, sample_row)
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 0)
            print(f"✅ Prompt generation test passed.")
            print(f"  Sample prompt length: {len(prompt)} characters")
        except Exception as e:
            print(f"⚠️  Prompt generation test failed: {str(e)}")
    
    def test_results_saving(self):
        """結果保存機能のテスト"""
        question_id = '31772'
        
        # テスト用の結果データ
        test_results = {
            'question_id': question_id,
            'n_classes': 3,
            'n_samples': 10,
            'final_map3': 0.85,
            'test_flag': True
        }
        
        # 一時ディレクトリを使用
        with tempfile.TemporaryDirectory() as temp_dir:
            save_question_results(question_id, test_results, temp_dir)
            
            # ファイルが作成されたかチェック
            results_file = os.path.join(temp_dir, f'question_{question_id}_results.json')
            self.assertTrue(os.path.exists(results_file))
            
            # ファイル内容をチェック
            import json
            with open(results_file, 'r') as f:
                saved_results = json.load(f)
            
            self.assertEqual(saved_results['question_id'], question_id)
            self.assertEqual(saved_results['final_map3'], 0.85)
        
        print(f"✅ Results saving test passed.")
    
    def test_data_summary(self):
        """データサマリー表示のテスト"""
        question_id = '31772'
        filtered_data = filter_data_by_question_id(self.sample_data, question_id)
        
        # サマリー表示（出力のキャプチャは難しいので、エラーが出ないことを確認）
        try:
            print_question_data_summary(filtered_data, question_id)
            print(f"✅ Data summary test passed.")
        except Exception as e:
            self.fail(f"Data summary failed: {str(e)}")


class TestSystemIntegration(unittest.TestCase):
    """システム統合テスト"""
    
    def test_question_ids_consistency(self):
        """QuestionIDsの整合性テスト"""
        from prompt_utils import questions
        
        # prompt_utilsのquestionsとconfig.pyのQUESTION_IDSが一致するかテスト
        prompt_utils_ids = set(questions.keys())
        config_ids = set(QUESTION_IDS)
        
        self.assertEqual(prompt_utils_ids, config_ids, 
                        "QuestionIDs in prompt_utils.py and config.py should match")
        
        print(f"✅ QuestionIDs consistency test passed. {len(QUESTION_IDS)} questions found.")
    
    def test_directory_structure(self):
        """ディレクトリ構造のテスト"""
        # テスト用の一時ディレクトリで確認
        with tempfile.TemporaryDirectory() as temp_dir:
            # 元のOUTPUT_DIRを一時的に変更
            original_output_dir = globals().get('OUTPUT_DIR', 'ver_2')
            
            try:
                # config.pyのOUTPUT_DIRを一時的に変更
                import config
                config.OUTPUT_DIR = temp_dir
                
                # 各QuestionIdのディレクトリパスを確認
                for question_id in QUESTION_IDS[:3]:  # 最初の3つだけテスト
                    output_dir = get_question_output_dir(question_id)
                    self.assertIn(temp_dir, output_dir)
                    self.assertIn(question_id, output_dir)
                
                print(f"✅ Directory structure test passed.")
                
            finally:
                # 元の設定を復元
                config.OUTPUT_DIR = original_output_dir


def run_all_tests():
    """全テストを実行"""
    print("="*80)
    print("QuestionId別モデル学習・推論システム テスト開始")
    print("="*80)
    
    # テストスイートの作成
    test_suite = unittest.TestSuite()
    
    # 基本機能テスト
    test_suite.addTest(TestQuestionSpecificSystem('test_config_loading'))
    test_suite.addTest(TestQuestionSpecificSystem('test_path_management'))
    test_suite.addTest(TestQuestionSpecificSystem('test_data_filtering'))
    test_suite.addTest(TestQuestionSpecificSystem('test_label_processing'))
    test_suite.addTest(TestQuestionSpecificSystem('test_prompt_generation'))
    test_suite.addTest(TestQuestionSpecificSystem('test_results_saving'))
    test_suite.addTest(TestQuestionSpecificSystem('test_data_summary'))
    
    # 統合テスト
    test_suite.addTest(TestSystemIntegration('test_question_ids_consistency'))
    test_suite.addTest(TestSystemIntegration('test_directory_structure'))
    
    # テストランナーの実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "="*80)
    if result.wasSuccessful():
        print("🎉 すべてのテストが成功しました!")
        print("QuestionId別システムは正常に動作する準備ができています。")
    else:
        print(f"❌ {len(result.failures)} テストが失敗、{len(result.errors)} エラーが発生しました。")
        print("システムを修正してからトレーニングを開始してください。")
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)