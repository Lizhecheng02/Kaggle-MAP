"""
データ拡張とクラスバランシング用ユーティリティ
"""

import pandas as pd
import numpy as np
from collections import Counter
import random
import re


class DataAugmenter:
    """データ拡張クラス"""
    
    def __init__(self, train_df):
        self.train_df = train_df
        self.misconception_examples = self._create_misconception_dict()
        
    def _create_misconception_dict(self):
        """各Misconceptionの例文を収集"""
        misc_dict = {}
        for misc in self.train_df['Misconception'].unique():
            if misc != 'NA':
                examples = self.train_df[self.train_df['Misconception'] == misc]
                misc_dict[misc] = examples
        return misc_dict
    
    def augment_with_paraphrasing(self, row):
        """説明文のパラフレーズによる拡張"""
        explanation = row['StudentExplanation']
        augmented_explanations = []
        
        # 数式や数値を保持しながら文章構造を変更
        patterns = [
            # 理由の言い換え
            (r"because", ["since", "as", "due to the fact that"]),
            (r"therefore", ["thus", "hence", "so"]),
            (r"however", ["but", "yet", "although"]),
            
            # 数学用語の同義語
            (r"bigger than", ["larger than", "greater than"]),
            (r"smaller than", ["less than", "lower than"]),
            (r"equals", ["is equal to", "is the same as"]),
            
            # 文構造の変更
            (r"I think", ["In my opinion", "I believe", "It seems to me"]),
            (r"This is", ["That is", "It is"]),
        ]
        
        # パラフレーズ生成
        for _ in range(2):  # 各サンプルから2つの拡張版を生成
            new_explanation = explanation
            for pattern, replacements in patterns:
                if re.search(pattern, new_explanation, re.IGNORECASE):
                    replacement = random.choice(replacements)
                    new_explanation = re.sub(pattern, replacement, new_explanation, flags=re.IGNORECASE)
            
            if new_explanation != explanation:
                augmented_explanations.append(new_explanation)
        
        return augmented_explanations
    
    def augment_minority_classes(self, min_samples=500):
        """少数クラスのオーバーサンプリング"""
        target_counts = self.train_df['target'].value_counts()
        augmented_rows = []
        
        for target, count in target_counts.items():
            if count < min_samples:
                # このターゲットのサンプルを取得
                target_samples = self.train_df[self.train_df['target'] == target]
                
                # 必要な追加サンプル数
                needed = min_samples - count
                
                for _ in range(needed):
                    # ランダムにサンプルを選択
                    sample = target_samples.sample(1).iloc[0].copy()
                    
                    # 説明文を拡張
                    augmented_explanations = self.augment_with_paraphrasing(sample)
                    
                    if augmented_explanations:
                        sample['StudentExplanation'] = augmented_explanations[0]
                        augmented_rows.append(sample)
        
        return pd.DataFrame(augmented_rows)
    
    def create_synthetic_misconceptions(self):
        """類似のMisconceptionパターンから合成データを生成"""
        synthetic_rows = []
        
        # 類似のMisconceptionパターンを定義
        misconception_patterns = {
            'Additive': ['Adding_across', 'Tacking'],
            'Subtraction': ['Inversion', 'Wrong_term'],
            'Wrong_fraction': ['Wrong_Fraction', 'Denominator-only_change'],
        }
        
        for base_misc, similar_miscs in misconception_patterns.items():
            if base_misc not in self.misconception_examples:
                continue
                
            base_examples = self.misconception_examples[base_misc]
            
            for similar_misc in similar_miscs:
                if similar_misc not in self.misconception_examples:
                    continue
                    
                similar_examples = self.misconception_examples[similar_misc]
                
                # 基本パターンと類似パターンを組み合わせて新しい例を生成
                for _ in range(min(10, len(base_examples), len(similar_examples))):
                    base_sample = base_examples.sample(1).iloc[0]
                    similar_sample = similar_examples.sample(1).iloc[0]
                    
                    # 説明文の一部を組み合わせ
                    base_parts = base_sample['StudentExplanation'].split('.')
                    similar_parts = similar_sample['StudentExplanation'].split('.')
                    
                    if len(base_parts) > 1 and len(similar_parts) > 1:
                        # 文の前半と後半を組み合わせ
                        new_explanation = base_parts[0] + ". " + similar_parts[-1]
                        
                        new_row = base_sample.copy()
                        new_row['StudentExplanation'] = new_explanation
                        synthetic_rows.append(new_row)
        
        return pd.DataFrame(synthetic_rows)
    
    def apply_all_augmentations(self):
        """すべての拡張手法を適用"""
        # 少数クラスの拡張
        minority_augmented = self.augment_minority_classes()
        
        # 合成データの生成
        synthetic_data = self.create_synthetic_misconceptions()
        
        # すべてを結合
        augmented_train = pd.concat([
            self.train_df,
            minority_augmented,
            synthetic_data
        ], ignore_index=True)
        
        return augmented_train


def apply_mixup(batch1, batch2, alpha=0.2):
    """
    Mixupデータ拡張の適用
    入力の線形補間を行う
    """
    lam = np.random.beta(alpha, alpha)
    
    mixed_inputs = {}
    mixed_inputs['input_ids'] = batch1['input_ids']  # トークンIDは混合しない
    mixed_inputs['attention_mask'] = batch1['attention_mask']
    
    # ラベルの混合（ソフトターゲット）
    if 'labels' in batch1:
        batch_size = len(batch1['labels'])
        num_classes = batch1['labels'].max() + 1
        
        labels1 = np.eye(num_classes)[batch1['labels']]
        labels2 = np.eye(num_classes)[batch2['labels']]
        
        mixed_labels = lam * labels1 + (1 - lam) * labels2
        mixed_inputs['labels'] = mixed_labels
    
    return mixed_inputs, lam