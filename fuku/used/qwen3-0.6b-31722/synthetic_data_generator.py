"""
合成データ生成スクリプト - Qwen3-0.6Bモデルの弱点を補強
"""

import random
import json
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class SyntheticDataGenerator:
    """合成データ生成クラス"""
    
    def __init__(self):
        # Neither用のテンプレート
        self.neither_templates = [
            "I think it's {answer} but I'm not sure",
            "Maybe {answer}?",
            "I don't really know but {answer}",
            "I just guessed because it looked right",
            "I forgot the steps",
            "{partial}... wait, that might be wrong",
            "The teacher said we do it this way, so {answer}",
            "I can't explain it, it just makes sense to me",
            "Not certain—picked {answer} because it seemed closest",
        ]
        
        # 簡潔な正解用テンプレート
        self.concise_correct_templates = [
            "{answer}",
            "{calculation} = {answer}",
            "Simplified to {answer}",
            "{answer} (calculation omitted)",
            "{step1} → {answer}",
            "Answer: {answer}",
            "The fraction simplifies to {answer}",
            "Count and simplify to get {answer}",
        ]
        
        # 不確実性を示す単語
        self.uncertainty_words = ["maybe", "probably", "I think", "I guess", "not sure", 
                                 "possibly", "might be", "could be", "I believe", "sort of",
                                 "kinda", "I feel like"]
        
        # 誤概念パターン
        self.misconception_patterns = {
            'sign_error': self._apply_sign_error,
            'operation_confusion': self._apply_operation_confusion,
            'incomplete': self._make_incomplete,
            'wrong_term': self._apply_wrong_term,
        }
    
    def generate_neither_data(self, correct_explanations: List[Dict], start_row_id: int = 0) -> List[Dict]:
        """Neitherカテゴリの合成データを生成"""
        synthetic_data = []
        row_id = start_row_id
        
        for item in correct_explanations:
            # 方法1: ノイズを追加して内容を破壊
            noisy = self._add_noise_truncation(item['explanation'])
            synthetic_data.append({
                'row_id': row_id,
                'QuestionId': item.get('question_id', random.randint(10000, 99999)),
                'QuestionText': item['question'],
                'MC_Answer': item['answer'],
                'StudentExplanation': noisy,
                'Category': 'True_Neither',
                'Misconception': 'NA'
            })
            row_id += 1
            
            # 方法2: 無関係な内容を挿入
            irrelevant = self._add_irrelevant_content(item['explanation'])
            synthetic_data.append({
                'row_id': row_id,
                'QuestionId': item.get('question_id', random.randint(10000, 99999)),
                'QuestionText': item['question'],
                'MC_Answer': item['answer'],
                'StudentExplanation': irrelevant,
                'Category': 'True_Neither',
                'Misconception': 'NA'
            })
            row_id += 1
            
            # 方法3: 曖昧なテンプレートを使用
            template = random.choice(self.neither_templates)
            vague = template.format(
                answer=item['answer'],
                partial=item['explanation'][:20]
            )
            synthetic_data.append({
                'row_id': row_id,
                'QuestionId': item.get('question_id', random.randint(10000, 99999)),
                'QuestionText': item['question'],
                'MC_Answer': item['answer'],
                'StudentExplanation': vague,
                'Category': 'True_Neither',
                'Misconception': 'NA'
            })
            row_id += 1
        
        return synthetic_data
    
    def generate_concise_correct_data(self, correct_explanations: List[Dict], start_row_id: int = 0) -> List[Dict]:
        """簡潔だが正しい説明の合成データを生成"""
        synthetic_data = []
        row_id = start_row_id
        
        for item in correct_explanations:
            # 方法1: キー計算のみを抽出
            key_calc = self._extract_key_calculation(item['explanation'])
            if key_calc:
                synthetic_data.append({
                    'row_id': row_id,
                    'QuestionId': item.get('question_id', random.randint(10000, 99999)),
                    'QuestionText': item['question'],
                    'MC_Answer': item['answer'],
                    'StudentExplanation': key_calc,
                    'Category': 'True_Correct',
                    'Misconception': 'NA'
                })
                row_id += 1
            
            # 方法2: テンプレートベースの簡潔化
            template = random.choice(self.concise_correct_templates)
            concise = template.format(
                answer=item['answer'],
                calculation=self._extract_calculation(item['explanation']),
                step1=self._get_first_step(item['explanation'])
            )
            synthetic_data.append({
                'row_id': row_id,
                'QuestionId': item.get('question_id', random.randint(10000, 99999)),
                'QuestionText': item['question'],
                'MC_Answer': item['answer'],
                'StudentExplanation': concise,
                'Category': 'True_Correct',
                'Misconception': 'NA'
            })
            row_id += 1
            
            # 方法3: 説明を極限まで短縮
            ultra_short = f"{item['answer']} is the answer"
            synthetic_data.append({
                'row_id': row_id,
                'QuestionId': item.get('question_id', random.randint(10000, 99999)),
                'QuestionText': item['question'],
                'MC_Answer': item['answer'],
                'StudentExplanation': ultra_short,
                'Category': 'True_Correct',
                'Misconception': 'NA'
            })
            row_id += 1
        
        return synthetic_data
    
    def generate_misconception_data(self, correct_explanations: List[Dict], start_row_id: int = 0) -> List[Dict]:
        """細かい誤概念の合成データを生成"""
        synthetic_data = []
        row_id = start_row_id
        
        misconception_types = {
            'Positive': 'sign_error',
            'SwapDividend': 'operation_confusion',
            'Incomplete': 'incomplete',
            'Wrong_term': 'wrong_term'
        }
        
        for item in correct_explanations:
            for misc_label, pattern_name in misconception_types.items():
                if pattern_name in self.misconception_patterns:
                    modified = self.misconception_patterns[pattern_name](item['explanation'])
                    if modified != item['explanation']:  # 変更があった場合のみ
                        synthetic_data.append({
                            'row_id': row_id,
                            'QuestionId': item.get('question_id', random.randint(10000, 99999)),
                            'QuestionText': item['question'],
                            'MC_Answer': item['answer'],
                            'StudentExplanation': modified,
                            'Category': 'True_Misconception',
                            'Misconception': misc_label
                        })
                        row_id += 1
        
        return synthetic_data
    
    def generate_contrastive_triplets(self, item: Dict, start_row_id: int = 0) -> List[Dict]:
        """対照的な3つ組（正解・誤概念・Neither）を生成"""
        triplet = []
        row_id = start_row_id
        question_id = item.get('question_id', random.randint(10000, 99999))
        
        # 1. 正解（オリジナル）
        triplet.append({
            'row_id': row_id,
            'QuestionId': question_id,
            'QuestionText': item['question'],
            'MC_Answer': item['answer'],
            'StudentExplanation': item['explanation'],
            'Category': 'True_Correct',
            'Misconception': 'NA'
        })
        row_id += 1
        
        # 2. 誤概念バージョン
        misconception = self._apply_random_misconception(item['explanation'])
        triplet.append({
            'row_id': row_id,
            'QuestionId': question_id,
            'QuestionText': item['question'],
            'MC_Answer': item['answer'],
            'StudentExplanation': misconception,
            'Category': 'True_Misconception',
            'Misconception': 'Synthetic'
        })
        row_id += 1
        
        # 3. Neitherバージョン
        neither = self._make_neither_from_correct(item['explanation'])
        triplet.append({
            'row_id': row_id,
            'QuestionId': question_id,
            'QuestionText': item['question'],
            'MC_Answer': item['answer'],
            'StudentExplanation': neither,
            'Category': 'True_Neither',
            'Misconception': 'NA'
        })
        row_id += 1
        
        return triplet
    
    # ヘルパーメソッド
    def _add_noise_truncation(self, text: str, p_drop: float = 0.6) -> str:
        """テキストにノイズを追加して内容を破壊"""
        words = text.split()
        keep = [w for w in words if random.random() > p_drop]
        random.shuffle(keep)
        
        # ランダムに無関係な文を挿入
        insertions = ["btw", "oh wait", "umm", "what was it again", "idk", "forgot"]
        insertion = random.choice(insertions)
        
        return " ".join(keep[:max(3, len(keep)//2)]) + f" {insertion}"
    
    def _add_irrelevant_content(self, text: str) -> str:
        """無関係な内容を追加"""
        irrelevant = [
            "Yesterday's homework was hard",
            "I think I've seen this problem before",
            "Will this be on the test?",
            "Can I use a calculator?",
            "My friend said something different",
            "The textbook had a similar example",
        ]
        return text[:30] + "... " + random.choice(irrelevant)
    
    def _extract_key_calculation(self, text: str) -> str:
        """キー計算を抽出"""
        # 数式パターンを探す
        math_pattern = r'(\d+[\s\+\-\*/=]+\d+[\s\+\-\*/=]*\d*)'
        matches = re.findall(math_pattern, text)
        return matches[0] if matches else ""
    
    def _extract_calculation(self, text: str) -> str:
        """計算部分を抽出"""
        calc_keywords = ["calculate", "equation", "=", "→", "equals", "gives"]
        for keyword in calc_keywords:
            if keyword in text:
                idx = text.find(keyword)
                return text[max(0, idx-10):idx+20].strip()
        return ""
    
    def _get_first_step(self, text: str) -> str:
        """最初のステップを取得"""
        sentences = text.split(".")
        return sentences[0] if sentences else text[:30]
    
    def _apply_sign_error(self, text: str) -> str:
        """符号エラーを適用"""
        # +を-に、-を+に変換
        text = re.sub(r'(?<!\d)\+(?!\d)', '###MINUS###', text)
        text = re.sub(r'(?<!\d)-(?!\d)', '+', text)
        text = text.replace('###MINUS###', '-')
        return text
    
    def _apply_operation_confusion(self, text: str) -> str:
        """演算の混同を適用"""
        replacements = [
            ("multiply", "divide"),
            ("times", "divided by"),
            ("×", "÷"),
            ("add", "subtract"),
            ("plus", "minus"),
            ("sum", "difference"),
        ]
        for old, new in replacements:
            if old in text:
                return text.replace(old, new)
        return text
    
    def _make_incomplete(self, text: str) -> str:
        """説明を不完全にする"""
        # 後半を削除
        words = text.split()
        cutoff = len(words) // 2
        return " ".join(words[:cutoff]) + "..."
    
    def _apply_wrong_term(self, text: str) -> str:
        """誤った項を適用"""
        replacements = [
            ("numerator", "denominator"),
            ("denominator", "numerator"),
            ("shaded", "unshaded"),
            ("white", "black"),
            ("top", "bottom"),
            ("not shaded", "shaded"),
        ]
        for old, new in replacements:
            if old in text:
                return text.replace(old, new)
        return text
    
    def _apply_random_misconception(self, text: str) -> str:
        """ランダムな誤概念を適用"""
        pattern = random.choice(list(self.misconception_patterns.values()))
        return pattern(text)
    
    def _make_neither_from_correct(self, text: str) -> str:
        """正解からNeitherを作成"""
        # キーステップを削除して曖昧にする
        uncertainty = random.choice(self.uncertainty_words)
        return text[:20] + f"... {uncertainty}"
    


def main():
    """使用例"""
    # サンプルデータ
    sample_data = [
        {
            'question': "What fraction of the shape is not shaded? Give your answer in its simplest form. [Image: A triangle split into 9 equal smaller triangles. 6 of them are shaded.]",
            'answer': r"\( \frac{1}{3} \)",
            'explanation': "6 out of 9 triangles are shaded, so 3 are not shaded. 3/9 simplifies to 1/3",
            'question_id': 31772
        },
        {
            'question': "Simplify 12/18",
            'answer': r"\( \frac{2}{3} \)",
            'explanation': "The GCD of 12 and 18 is 6, so divide both by 6: 12÷6=2, 18÷6=3, answer is 2/3",
            'question_id': 31773
        }
    ]
    
    generator = SyntheticDataGenerator()
    
    # 各種合成データを生成（row_idを連続で管理）
    row_id_counter = 0
    neither_data = generator.generate_neither_data(sample_data, start_row_id=row_id_counter)
    row_id_counter += len(neither_data)
    
    concise_data = generator.generate_concise_correct_data(sample_data, start_row_id=row_id_counter)
    row_id_counter += len(concise_data)
    
    misconception_data = generator.generate_misconception_data(sample_data, start_row_id=row_id_counter)
    
    # 結果を表示
    print("=== Neither Data ===")
    for item in neither_data[:3]:
        print(f"Category: {item['Category']}")
        print(f"Misconception: {item['Misconception']}")
        print(f"StudentExplanation: {item['StudentExplanation'][:100]}...")
        print()
    
    print("\n=== Concise Correct Data ===")
    for item in concise_data[:3]:
        print(f"Category: {item['Category']}")
        print(f"Misconception: {item['Misconception']}")
        print(f"StudentExplanation: {item['StudentExplanation'][:100]}...")
        print()
    
    print("\n=== Misconception Data ===")
    for item in misconception_data[:3]:
        print(f"Category: {item['Category']}")
        print(f"Misconception: {item['Misconception']}")
        print(f"StudentExplanation: {item['StudentExplanation'][:100]}...")
        print()
    
    # 全データを保存（train.csvと同じ形式）
    all_synthetic = neither_data + concise_data + misconception_data
    df = pd.DataFrame(all_synthetic)
    # train.csvと同じ列順にする
    df = df[['row_id', 'QuestionId', 'QuestionText', 'MC_Answer', 'StudentExplanation', 'Category', 'Misconception']]
    df.to_csv('synthetic_training_data.csv', index=False)
    print(f"\n合成データ {len(all_synthetic)} 件を synthetic_training_data.csv に保存しました")


if __name__ == "__main__":
    main()