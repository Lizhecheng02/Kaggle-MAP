"""
DeepSeek-R1-Distill-Qwen-32B設定テスト
"""
import unittest
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *


class TestConfig(unittest.TestCase):
    """設定値のテスト"""
    
    def test_model_configuration(self):
        """モデル設定のテスト"""
        self.assertEqual(VER, "bnb4bit")
        self.assertEqual(MODEL_NAME, "/hdd/models/DeepSeek-R1-Distill-Qwen-32B")
        self.assertEqual(MODEL_TYPE, "qwen2")
        self.assertIsInstance(EPOCHS, int)
        self.assertGreater(EPOCHS, 0)
        self.assertIsInstance(MAX_LEN, int)
        self.assertGreater(MAX_LEN, 0)
    
    def test_training_parameters(self):
        """トレーニングパラメータのテスト"""
        self.assertIsInstance(TRAIN_BATCH_SIZE, int)
        self.assertGreater(TRAIN_BATCH_SIZE, 0)
        self.assertIsInstance(EVAL_BATCH_SIZE, int)
        self.assertGreater(EVAL_BATCH_SIZE, 0)
        self.assertIsInstance(GRADIENT_ACCUMULATION_STEPS, int)
        self.assertGreater(GRADIENT_ACCUMULATION_STEPS, 0)
        self.assertIsInstance(LEARNING_RATE, float)
        self.assertGreater(LEARNING_RATE, 0)
    
    def test_lora_configuration(self):
        """LoRA設定のテスト"""
        self.assertEqual(LORA_RANK, 128)
        self.assertEqual(LORA_ALPHA, 256)
        self.assertIsInstance(LORA_TARGET_MODULES, list)
        self.assertIn("q_proj", LORA_TARGET_MODULES)
        self.assertIn("v_proj", LORA_TARGET_MODULES)
        self.assertIsInstance(LORA_DROPOUT, float)
        self.assertGreaterEqual(LORA_DROPOUT, 0)
        self.assertLessEqual(LORA_DROPOUT, 1)
        self.assertEqual(LORA_BIAS, "none")
    
    def test_bitsandbytes_configuration(self):
        """BitsAndBytes設定のテスト"""
        self.assertTrue(USE_4BIT_QUANTIZATION)
        self.assertEqual(BNB_4BIT_COMPUTE_DTYPE, "bfloat16")
        self.assertEqual(BNB_4BIT_QUANT_TYPE, "nf4")
        self.assertTrue(BNB_4BIT_USE_DOUBLE_QUANT)
    
    def test_paths(self):
        """パス設定のテスト"""
        self.assertEqual(OUTPUT_DIR, f"ver_{VER}")
        self.assertEqual(BEST_MODEL_PATH, f"{OUTPUT_DIR}/best_map3")
        self.assertEqual(LABEL_ENCODER_PATH, f"{OUTPUT_DIR}/label_encoder.joblib")
        
    def test_wandb_settings(self):
        """WandB設定のテスト"""
        self.assertIsInstance(USE_WANDB, bool)
        self.assertIn("deepseek-r1-32b-bnb4bit", WANDB_PROJECT)
        self.assertIn("deepseek-r1-32b-bnb4bit", WANDB_RUN_NAME)


if __name__ == "__main__":
    unittest.main()