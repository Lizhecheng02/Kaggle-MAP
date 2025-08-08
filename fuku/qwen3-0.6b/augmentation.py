"""
Data Augmentation module for math misconception dataset
"""

import random
import re
import pandas as pd
from typing import List, Dict, Tuple
import numpy as np


class MathTextAugmenter:
    """数学問題の生徒回答テキストに対するData Augmentation"""
    
    def __init__(self, augment_prob: float = 0.5, seed: int = 42):
        """
        Args:
            augment_prob: 各Augmentation手法を適用する確率
            seed: ランダムシード
        """
        self.augment_prob = augment_prob
        random.seed(seed)
        np.random.seed(seed)
        
        # 数学用語の同義語辞書
        self.math_synonyms = {
            "divided by": ["÷", "divided", "over", "/"],
            "multiplied by": ["×", "times", "multiplied", "*"],
            "plus": ["+", "added to", "and"],
            "minus": ["-", "subtract", "less"],
            "equals": ["=", "is", "equal to", "is equal to"],
            "fraction": ["ratio", "part"],
            "shaded": ["colored", "filled", "marked"],
            "not shaded": ["unshaded", "not colored", "blank", "empty"],
            "triangle": ["triangular shape", "triangles"],
            "simplified": ["in simplest form", "reduced", "simplest"],
            "because": ["since", "as", "due to"],
            "therefore": ["so", "thus", "hence"],
        }
        
        # 数式表記の変換パターン
        self.fraction_patterns = {
            r"\\frac\{(\d+)\}\{(\d+)\}": r"\1/\2",  # LaTeX分数 → スラッシュ表記
            r"(\d+)/(\d+)": r"\\frac{\1}{\2}",  # スラッシュ表記 → LaTeX分数
            r"(\d+) over (\d+)": r"\1/\2",  # テキスト表記 → スラッシュ表記
        }
        
        # タイポのパターン（実際の生徒の回答に見られるもの）
        self.typo_patterns = [
            ("shaded", ["shadedd", "shaded", "shadd"]),
            ("simplified", ["simplafied", "simplifed", "simplafide"]),
            ("equivalent", ["equivilent", "equivalant", "equivelent"]),
            ("divided", ["devided", "divded"]),
            ("because", ["becuase", "becaus", "bcause"]),
            ("therefore", ["therfore", "therefor"]),
            ("third", ["thrid", "3rd"]),
            ("ninth", ["nineth", "9th"]),
        ]

    def augment_text(self, text: str, explanation: str, category: str) -> List[Dict[str, str]]:
        """
        テキストと説明文をAugmentationして複数のバリエーションを生成
        
        Args:
            text: 元のテキスト（問題文 + 回答 + 説明文）
            explanation: 生徒の説明文
            category: True_Correct, True_Neither, etc.
            
        Returns:
            Augmentedされたテキストのリスト
        """
        augmented_samples = []
        
        # 1. 同義語置換
        if random.random() < self.augment_prob:
            aug_text = self._synonym_replacement(text)
            if aug_text != text:
                augmented_samples.append({"text": aug_text, "type": "synonym"})
        
        # 2. 数式表記の変換
        if random.random() < self.augment_prob:
            aug_text = self._math_notation_conversion(text)
            if aug_text != text:
                augmented_samples.append({"text": aug_text, "type": "notation"})
        
        # 3. タイポの追加（True_Correctの場合のみ、精度を保つため）
        if category == "True_Correct" and random.random() < self.augment_prob * 0.3:
            aug_text = self._add_typos(text)
            if aug_text != text:
                augmented_samples.append({"text": aug_text, "type": "typo"})
        
        # 4. パラフレージング（簡単なパターンベース）
        if random.random() < self.augment_prob:
            aug_text = self._simple_paraphrase(text, explanation)
            if aug_text != text:
                augmented_samples.append({"text": aug_text, "type": "paraphrase"})
        
        return augmented_samples
    
    def _synonym_replacement(self, text: str) -> str:
        """同義語置換"""
        aug_text = text
        for original, synonyms in self.math_synonyms.items():
            if original.lower() in aug_text.lower() and random.random() < 0.3:
                synonym = random.choice(synonyms)
                # 大文字小文字を考慮した置換
                aug_text = re.sub(
                    re.escape(original), 
                    synonym, 
                    aug_text, 
                    flags=re.IGNORECASE
                )
        return aug_text
    
    def _math_notation_conversion(self, text: str) -> str:
        """数式表記の変換"""
        aug_text = text
        
        # LaTeX分数をスラッシュ表記に変換
        if r"\frac" in aug_text and random.random() < 0.5:
            aug_text = re.sub(r"\\frac\{(\d+)\}\{(\d+)\}", r"\1/\2", aug_text)
        # スラッシュ表記をテキスト表記に変換
        elif re.search(r"(\d+)/(\d+)", aug_text) and random.random() < 0.5:
            aug_text = re.sub(r"(\d+)/(\d+)", r"\1 over \2", aug_text)
        
        return aug_text
    
    def _add_typos(self, text: str) -> str:
        """実際の生徒の回答に見られるようなタイポを追加"""
        aug_text = text
        for correct, typos in self.typo_patterns:
            if correct in aug_text.lower() and random.random() < 0.2:
                typo = random.choice(typos)
                aug_text = re.sub(
                    re.escape(correct),
                    typo,
                    aug_text,
                    flags=re.IGNORECASE,
                    count=1
                )
        return aug_text
    
    def _simple_paraphrase(self, text: str, explanation: str) -> str:
        """簡単なパラフレージング"""
        # 説明文の順序を変更する簡単なパラフレージング
        if "because" in explanation.lower():
            parts = re.split(r"\bbecause\b", explanation, flags=re.IGNORECASE)
            if len(parts) == 2:
                # "A because B" → "Since B, A" のような変換
                paraphrased = f"Since {parts[1].strip()}, {parts[0].strip()}"
                aug_text = text.replace(explanation, paraphrased)
                return aug_text
        
        if "so" in explanation.lower():
            parts = re.split(r"\bso\b", explanation, flags=re.IGNORECASE)
            if len(parts) == 2:
                # "A so B" → "B because A" のような変換
                paraphrased = f"{parts[1].strip()} because {parts[0].strip()}"
                aug_text = text.replace(explanation, paraphrased)
                return aug_text
        
        return text


def augment_dataset(df: pd.DataFrame, augmenter: MathTextAugmenter, 
                    augment_ratio: float = 0.5) -> pd.DataFrame:
    """
    データセット全体にAugmentationを適用
    
    Args:
        df: 元のDataFrame
        augmenter: MathTextAugmenterインスタンス
        augment_ratio: Augmentationを適用するサンプルの割合
        
    Returns:
        Augmentedデータを含む新しいDataFrame
    """
    augmented_rows = []
    
    # サンプリングするデータの選択
    sample_size = int(len(df) * augment_ratio)
    sampled_df = df.sample(n=sample_size, random_state=42)
    
    for idx, row in sampled_df.iterrows():
        # 各行に対してAugmentationを実行
        augmented_texts = augmenter.augment_text(
            row['text'],
            row.get('StudentExplanation', ''),
            row.get('Category', 'True_Neither')
        )
        
        # Augmentedデータを新しい行として追加
        for aug_data in augmented_texts:
            new_row = row.copy()
            new_row['text'] = aug_data['text']
            new_row['augmentation_type'] = aug_data['type']
            new_row['is_augmented'] = True
            augmented_rows.append(new_row)
    
    # 元のデータにis_augmentedフラグを追加
    df['is_augmented'] = False
    df['augmentation_type'] = 'original'
    
    # Augmentedデータを結合
    if augmented_rows:
        augmented_df = pd.DataFrame(augmented_rows)
        combined_df = pd.concat([df, augmented_df], ignore_index=True)
        print(f"Added {len(augmented_rows)} augmented samples to {len(df)} original samples")
        return combined_df
    
    return df


if __name__ == "__main__":
    # テスト用のコード
    sample_text = """What fraction of the shape is not shaded? 
    Give your answer in its simplest form. 
    [Image: A triangle split into 9 equal smaller triangles. 6 of them are shaded.]
    \( \\frac{1}{3} \)
    1/3 because 6 over 9 is 2 thirds and 1 third is not shaded."""
    
    sample_explanation = "1/3 because 6 over 9 is 2 thirds and 1 third is not shaded."
    
    augmenter = MathTextAugmenter(augment_prob=1.0)
    augmented = augmenter.augment_text(sample_text, sample_explanation, "True_Correct")
    
    print("Original text:")
    print(sample_text)
    print("\nAugmented variations:")
    for i, aug in enumerate(augmented, 1):
        print(f"\n{i}. Type: {aug['type']}")
        print(aug['text'])