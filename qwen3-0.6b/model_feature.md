# Qwen3-0.6Bモデル特徴分析レポート

## 概要
本レポートでは、数学の誤概念分類タスクにおけるQwen3-0.6Bモデルの誤分類パターンを分析し、モデルの特徴と限界を明らかにする。

## 主要な誤分類パターン

### 1. 「Neither」カテゴリでの大規模な誤分類（最大の問題点）
- **エラー数**: 1,093件（True_Neither:NA 603件 + False_Neither:NA 490件）
- **特徴**: モデルは「正解でも既知の誤概念でもない」という否定的な定義を認識することが苦手

### 2. 正解認識の失敗
- **エラー数**: True_Correct:NA 409件
- **特徴**: 数学的に正しい回答であっても、モデルが正解として認識できない

### 3. 誤概念カテゴリ特有の混乱
- **主要なエラー**:
  - Incomplete（不完全）: 91件
  - Positive（符号）: 33件
  - Subtraction（引き算）: 30件
  - Wrong_term（項の誤り）: 25件

## モデルの根本的な特徴と弱点

### 1. 容量制限による表層的なパターンマッチング
**問題点**:
- 0.6Bという小規模パラメータでは、深い意味理解が困難
- キーワードの存在/不在に過度に依存した分類を行う傾向
- 文脈全体を考慮した判断ができない

**具体例**:
- 「3/9 = 1/3」という正しい簡約化の説明でも、説明が簡潔すぎると「不完全」と誤判定

### 2. 「Neither」カテゴリの認識困難
**原因**:
- 「Neither」は他のカテゴリの「不在」で定義される低エントロピーなクラス境界
- 肯定的なパターン検出より、否定的な定義の認識は認知的に困難
- 限られたアテンションヘッドでは、長距離の談話追跡が不可能

**影響**:
- モデルは常に何らかの肯定的カテゴリ（正解or誤概念）に分類しようとする

### 3. 数学的正確性より説明形式を重視
**問題点**:
- 数学的に正しい答えでも、説明が簡潔だと誤りと判定
- 記号的な数学パース能力が弱い
- 「説明の形式」と「数学的実質」を区別できない

**例**:
- 「1/3 is the simplest form」（正解）→ 説明不足として誤分類

### 4. 細かい誤概念の区別困難
**特に苦手な分野**:
- **Incomplete（不完全）**: 反事実的推論が必要（「学生が本当に理解していたら何を言うべきだったか」）
- **符号エラー/演算の混同**: 長い文章中の「-」「引く」「残り」などの語彙的手がかりを数値的意味と結びつけられない
- **Wrong_term**: 問題文の視覚的/空間的推論（「塗られた」vs「塗られていない」）が困難

## 容量関連の一般的弱点

1. **長文脈のコヒーレンス不足**: 複数文の学生回答でエラー率が上昇
2. **語彙的ショートカットへの過度の依存**: 誤概念に関連するキーワードが正しい説明内に現れると誤予測
3. **キャリブレーション不良**: 上位2つのラベル間のロジット差が小さく、閾値設定で判定が変わる
4. **クラス不均衡への感度**: マイノリティクラス（「Neither」、稀な誤概念）の再現率が著しく低い

## 改善提案

### 1. データとトレーニングの改善
- クラスバランシングまたはコスト感応損失の導入
- 「Neither」とマイノリティ誤概念により高い損失重みを設定

### 2. アーキテクチャの工夫
- **カスケード評価**:
  1. 第1段階：数学的正確性の判定（答えは正しいか？）
  2. 第2段階：説明の質の評価
- **ルールベース数学チェッカーとのアンサンブル**: シンボリックな分数評価器で数値的正確性を補完

### 3. 学習方法の改善
- **説明生成を伴う指示調整**: 「Neitherを選んだ理由は...」という説明を生成させることで、証拠の不在への注意を強制
- **知識蒸留**: より大きな教師モデル（7B/14B）からの蒸留、特に微妙な否定クラスに有効

## 結論

Qwen3-0.6Bは容量制限による典型的な表層手がかり駆動型分類器の特徴を示している：

- 「存在」ラベルを「不在」ラベル（Neither）より優先
- 簡潔さを誤りと混同し、数学的真理の検証が困難
- 符号、演算、欠落した推論の追跡を要する細かい誤概念を混同

これらの観察結果は、0.6Bパラメータ予算に内在するアテンションと推論の限界を克服するために、以下に焦点を当てた今後の作業を示唆している：
1. 「Neither」と「Incomplete」概念のための、よりリッチでバランスの取れたトレーニング信号
2. 補助的なシンボリックまたはルーブリック明示的な特徴
3. より大きなモデルまたは説明強制トレーニングの活用

# English Part

# Qwen3-0.6B Model Feature Analysis Report

## Summary
This report analyzes the misclassification patterns of the Qwen3-0.6B model in the student math misconception classification task and highlights its characteristics and limitations.

## Major Misclassification Patterns

### 1. Extensive Misclassification in the "Neither" Category (Biggest Issue)
- **Error Count**: 1,093 (True_Neither:NA 603 + False_Neither:NA 490)
- **Characteristic**: The model struggles to recognize the negative definition of "neither the correct answer nor a known misconception"

### 2. Failure to Recognize Correct Answers
- **Error Count**: True_Correct:NA 409
- **Characteristic**: Even mathematically correct answers are not recognized as correct by the model

### 3. Confusion Specific to Misconception Categories
- **Major Errors**:
  - Incomplete: 91
  - Positive (sign): 33
  - Subtraction: 30
  - Wrong_term: 25

## Fundamental Characteristics and Weaknesses of the Model

### 1. Surface-level Pattern Matching Due to Capacity Limitations
**Issues**:
- With only 0.6B parameters, deep semantic understanding is difficult
- Tends to overly rely on the presence or absence of keywords for classification
- Cannot make judgments considering the entire context

**Example**:
- A correct simplification explanation like "3/9 = 1/3" is misclassified as "Incomplete" when it is too concise

### 2. Difficulty Recognizing the "Neither" Category
**Causes**:
- "Neither" is defined as the absence of other categories, creating a low-entropy class boundary
- Recognizing negative definitions is cognitively more difficult than detecting positive patterns
- Limited attention heads cannot track long-range discourse

**Impact**:
- The model always tries to categorize into some positive class (correct answer or misconception)

### 3. Prioritizing Explanation Format Over Mathematical Accuracy
**Issues**:
- Mathematically correct answers are judged incorrect if explanations are too brief
- Weak symbolic math parsing ability
- Cannot distinguish between "explanation format" and "mathematical substance"

**Example**:
- "1/3 is the simplest form" (correct) → misclassified due to insufficient explanation

### 4. Difficulty Distinguishing Fine-Grained Misconceptions
**Particularly Weak Areas**:
- Incomplete: requires counterfactual reasoning ("what the student would have said if they truly understood")
- Sign errors/operation confusion: fails to link lexical cues like "-", "minus", or "remaining" in long sentences to numerical meaning
- Wrong_term: struggles with visual/spatial reasoning in problem statements ("shaded" vs "unshaded")

## General Weaknesses Related to Capacity
1. Lack of coherence in long contexts: error rate increases for multi-sentence student responses
2. Overreliance on lexical shortcuts: mispredicts when misconception-related keywords appear in correct explanations
3. Poor calibration: small logit differences between the top two labels lead to threshold-based judgment changes
4. Sensitivity to class imbalance: recall for minority classes (Neither, rare misconceptions) is significantly low

## Improvement Suggestions

### 1. Data and Training Enhancements
- Introduce class balancing or cost-sensitive loss
- Assign higher loss weights to "Neither" and minority misconception classes

### 2. Architectural Strategies
- **Cascade Evaluation**:
  1. Stage 1: Judge mathematical accuracy (Is the answer correct?)
  2. Stage 2: Evaluate explanation quality
- **Ensemble with Rule-based Math Checker**: complement numerical accuracy with a symbolic fraction evaluator

### 3. Training Method Improvements
- Instruction tuning with explanation generation: force attention to absence of evidence by generating prompts like "Reason for choosing Neither..."
- Knowledge distillation: distill from larger teacher models (7B/14B), especially effective for subtle negative classes

## Conclusion
Qwen3-0.6B exhibits characteristics of a typical surface clue–driven classifier due to capacity constraints:

- Prioritizes the "presence" label over the "absence" label (Neither)
- Confuses conciseness with error, making mathematical truth verification difficult
- Mixes up fine-grained misconceptions that require tracking signs, operations, or missing reasoning

These observations suggest future work focusing on:
1. More rich and balanced training signals for the "Neither" and "Incomplete" concepts
2. Supplemental symbolic or rubric-explicit features
3. Leveraging larger models or explanation-enforced training
