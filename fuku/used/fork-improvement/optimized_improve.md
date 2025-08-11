# XGBoostモデル改良版の変更内容 / XGBoost Model Improvements

## 1. 特徴量エンジニアリングの強化 / Feature Engineering Enhancement

### 数学的特徴の拡張 / Mathematical Feature Extension
- **分数検出**: `FRAC_\d+_\d+` パターンと `\frac` 記法の両方を検出
- **数値カウント**: 問題文、回答、説明文それぞれの数値の数
- **演算子カウント**: +, -, *, /, = などの演算子の出現回数
- **括弧カウント**: 数式の複雑さを示す指標として
- **小数点カウント**: 小数を含む問題の検出
- **変数カウント**: 単一文字の変数（a, b, x, y など）の検出
- **数学用語カウント**: equation, solve, fraction など17種類の数学用語

**English:**
- **Fraction detection**: Detects both `FRAC_\d+_\d+` patterns and `\frac` notation
- **Number count**: Counts numbers in question text, answers, and student explanations
- **Operator count**: Occurrence count of operators like +, -, *, /, =
- **Parenthesis count**: Indicator of mathematical expression complexity
- **Decimal count**: Detection of problems containing decimals
- **Variable count**: Detection of single-letter variables (a, b, x, y, etc.)
- **Math term count**: 17 types of mathematical terms including equation, solve, fraction

### 統計的特徴 / Statistical Features
- **単語数**: 説明文の単語数
- **平均単語長**: 説明の詳細度を測る指標
- **ユニーク単語率**: 語彙の多様性
- **文の数**: 説明の構造化度合い
- **大文字率**: 強調や数式表記の指標

**English:**
- **Word count**: Number of words in explanations
- **Average word length**: Metric for explanation detail level
- **Unique word ratio**: Vocabulary diversity
- **Sentence count**: Degree of explanation structure
- **Uppercase ratio**: Indicator of emphasis or mathematical notation

### エラーパターン特徴 / Error Pattern Features
- **加減法の混同**: add と subtract が両方含まれる場合
- **乗除法の混同**: multiply と divide が両方含まれる場合
- **演算順序の問題**: order, first, then などの単語の出現
- **分数・小数の混同**: fraction と decimal が両方含まれる
- **負数の問題**: negative や minus の出現

**English:**
- **Addition/subtraction confusion**: When both 'add' and 'subtract' are present
- **Multiplication/division confusion**: When both 'multiply' and 'divide' are present
- **Order of operations issues**: Presence of words like 'order', 'first', 'then'
- **Fraction/decimal confusion**: When both 'fraction' and 'decimal' are present
- **Negative number issues**: Presence of 'negative' or 'minus'

### 相互作用特徴 / Interaction Features
- **total_numbers**: 全体の数値の総数
- **total_operators**: 全体の演算子の総数
- **math_complexity**: 数値数 × 演算子数で計算される複雑さ指標

**English:**
- **total_numbers**: Total count of all numbers
- **total_operators**: Total count of all operators
- **math_complexity**: Complexity metric calculated as number count × operator count

## 2. XGBoostパラメータの最適化とアンサンブル / XGBoost Parameter Optimization and Ensemble

### アンサンブル戦略 / Ensemble Strategy
2つの異なる設定のXGBoostモデルを学習し、予測を平均化：

**English:**
Train two XGBoost models with different configurations and average their predictions:

**モデル1（深い木・低学習率）**:
- max_depth: 12（元: 8）
- learning_rate: 0.03（元: 0.05）
- subsample: 0.9
- colsample_bytree: 0.9
- colsample_bylevel: 0.6
- min_child_weight: 2
- gamma: 0.15
- alpha: 0.1
- lambda: 1.5

**モデル2（中程度の深さ・中学習率）**:
- max_depth: 10
- learning_rate: 0.05
- subsample: 0.85
- colsample_bytree: 0.85
- colsample_bylevel: 0.7
- min_child_weight: 3
- gamma: 0.1
- alpha: 0.2
- lambda: 1.0

### その他の改良 / Other Improvements
- num_boost_round: 1500（元: 1000）
- early_stopping_rounds: 100（元: 50）

**English:**
- num_boost_round: 1500 (from: 1000)
- early_stopping_rounds: 100 (from: 50)

## 3. TF-IDF特徴量の改良 / TF-IDF Feature Improvements

- **n-gram範囲**: (1, 4)に拡張（元: (1, 3)）
- **max_features**: 8000（元: 5000）
- **カスタムストップワード**: 一般的な英語のストップワードに加えて、'question', 'answer', 'explanation', 'student'を追加
- **sublinear_tf**: Trueに設定してTF-IDFのスケーリングを改善
- **max_df**: 0.9（元: 0.95）

**English:**
- **n-gram range**: Extended to (1, 4) (from: (1, 3))
- **max_features**: 8000 (from: 5000)
- **Custom stopwords**: Added 'question', 'answer', 'explanation', 'student' to standard English stopwords
- **sublinear_tf**: Set to True for improved TF-IDF scaling
- **max_df**: 0.9 (from: 0.95)

## 4. テキストクリーニングの高度化 / Advanced Text Cleaning

### advanced_clean関数の改良 / Improvements to advanced_clean Function
- 分数表記の正規化: `1/2` → `FRAC_1_2`
- LaTeX分数の検出: `\frac{1}{2}` → `FRAC_1_2`
- 演算子の単語化:
  - `+` → `PLUS`
  - `-` → `MINUS`
  - `*` → `TIMES`
  - `÷` → `DIVIDE`
  - `=` → `EQUALS`

**English:**
- Fraction notation normalization: `1/2` → `FRAC_1_2`
- LaTeX fraction detection: `\frac{1}{2}` → `FRAC_1_2`
- Operator verbalization:
  - `+` → `PLUS`
  - `-` → `MINUS`
  - `*` → `TIMES`
  - `÷` → `DIVIDE`
  - `=` → `EQUALS`

## 5. 後処理機能の追加 / Post-processing Functionality

### post_process_predictions関数 / post_process_predictions Function
- 問題の複雑さに基づいて予測の信頼度を調整する機能を実装
- 現在は簡単な問題（数値が2個以下）を識別するマスクを作成
- 将来的にカテゴリごとの頻度調整などの拡張が可能

**English:**
- Implements functionality to adjust prediction confidence based on problem complexity
- Currently creates masks to identify simple problems (2 or fewer numbers)
- Future extensions possible for category-based frequency adjustments

## 期待される効果 / Expected Effects

1. **より豊富な特徴量**: 数学的な誤解パターンをより細かく捉えることが可能
2. **モデルの多様性**: アンサンブルにより、単一モデルの過学習リスクを低減
3. **テキスト理解の向上**: 4-gramとカスタムストップワードにより、数学固有の表現をより良く捉える
4. **予測の安定性**: 後処理により、極端な予測を調整可能

これらの改良により、MAP@3スコアの向上が期待されます。

**English:**
1. **Richer features**: Enables finer capture of mathematical misconception patterns
2. **Model diversity**: Ensemble reduces overfitting risk of single models
3. **Improved text understanding**: 4-grams and custom stopwords better capture math-specific expressions
4. **Prediction stability**: Post-processing enables adjustment of extreme predictions

These improvements are expected to increase the MAP@3 score.