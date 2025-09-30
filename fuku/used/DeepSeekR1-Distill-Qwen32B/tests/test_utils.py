"""
ユーティリティ関数のテスト
"""
import unittest
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import prepare_correct_answers, format_input, compute_map3, create_submission


class TestUtils(unittest.TestCase):
    """ユーティリティ関数のテスト"""
    
    def setUp(self):
        """テストデータのセットアップ"""
        self.sample_train_data = pd.DataFrame({
            'Category': ['True_Category1', 'False_Category1', 'True_Category2'],
            'QuestionId': [1, 1, 2],
            'MC_Answer': ['A', 'B', 'A'],
            'Misconception': ['NA', 'Wrong_concept', 'NA']
        })
        
        self.sample_row = {
            'is_correct': 1,
            'QuestionText': 'What is 2 + 2?',
            'MC_Answer': 'A) 4',
            'StudentExplanation': 'I added the numbers together.'
        }
    
    def test_prepare_correct_answers(self):
        """正解答案データ準備のテスト"""
        correct = prepare_correct_answers(self.sample_train_data)
        
        # 結果の検証
        self.assertIsInstance(correct, pd.DataFrame)
        self.assertIn('QuestionId', correct.columns)
        self.assertIn('MC_Answer', correct.columns)
        self.assertIn('is_correct', correct.columns)
        
        # 正解データのみが含まれることを確認
        self.assertTrue(all(correct['is_correct'] == 1))
    
    def test_format_input_correct(self):
        """正解の場合のプロンプトフォーマット"""
        prompt = format_input(self.sample_row)
        
        # 期待される要素が含まれているかチェック
        self.assertIn("Mathematical Misconception Analysis Task", prompt)
        self.assertIn("Question: What is 2 + 2?", prompt)
        self.assertIn("Answer: A) 4", prompt)
        self.assertIn("Correct?: Yes", prompt)
        self.assertIn("Explanation: I added the numbers together.", prompt)
        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("<|im_end|>", prompt)
        self.assertIn("<|im_start|>assistant", prompt)
        self.assertIn("<think>", prompt)
        self.assertIn("</think>", prompt)
    
    def test_format_input_incorrect(self):
        """不正解の場合のプロンプトフォーマット"""
        incorrect_row = self.sample_row.copy()
        incorrect_row['is_correct'] = 0
        
        prompt = format_input(incorrect_row)
        self.assertIn("Correct?: No", prompt)
    
    def test_compute_map3(self):
        """MAP@3計算のテスト"""
        # モックデータの作成
        logits = np.array([[0.1, 0.9, 0.0], [0.3, 0.2, 0.5], [0.4, 0.4, 0.2]])
        labels = np.array([1, 2, 0])
        
        eval_pred = Mock()
        eval_pred.predictions = logits
        eval_pred.labels = labels
        
        # 既存のeval_predオブジェクトを使用できるように修正
        result = compute_map3((logits, labels))
        
        self.assertIn("map@3", result)
        self.assertIsInstance(result["map@3"], float)
        self.assertGreaterEqual(result["map@3"], 0.0)
        self.assertLessEqual(result["map@3"], 1.0)
    
    def test_create_submission(self):
        """提出ファイル作成のテスト"""
        # モックデータの作成
        predictions = Mock()
        predictions.predictions = np.array([[0.1, 0.9, 0.0], [0.3, 0.2, 0.5]])
        
        test_data = pd.DataFrame({
            'row_id': [1, 2]
        })
        
        # モックラベルエンコーダー
        label_encoder = Mock()
        label_encoder.inverse_transform.return_value = np.array([
            'Category1:Misconception1', 'Category2:Misconception2', 
            'Category3:Misconception3', 'Category1:Misconception4',
            'Category2:Misconception5', 'Category3:Misconception6'
        ])
        
        submission = create_submission(predictions, test_data, label_encoder)
        
        # 結果の検証
        self.assertIsInstance(submission, pd.DataFrame)
        self.assertIn('row_id', submission.columns)
        self.assertIn('Category:Misconception', submission.columns)
        self.assertEqual(len(submission), 2)


class TestPromptQuality(unittest.TestCase):
    """プロンプト品質のテスト"""
    
    def test_prompt_structure(self):
        """プロンプト構造の妥当性テスト"""
        sample_row = {
            'is_correct': 1,
            'QuestionText': 'Test question',
            'MC_Answer': 'Test answer',
            'StudentExplanation': 'Test explanation'
        }
        
        prompt = format_input(sample_row)
        
        # DeepSeek-R1の形式に適合するかチェック
        self.assertTrue(prompt.startswith("<|im_start|>user"))
        self.assertIn("<|im_end|>", prompt)
        self.assertIn("<|im_start|>assistant", prompt)
        
        # CoT推論用のthinkタグが含まれているか
        self.assertIn("<think>", prompt)
        self.assertIn("</think>", prompt)
        
        # 推論ステップが含まれているか
        self.assertIn("step by step", prompt)
        self.assertIn("Understanding the question", prompt)
        self.assertIn("Evaluating the student's answer", prompt)
        self.assertIn("Identifying the underlying misconception", prompt)


if __name__ == "__main__":
    unittest.main()