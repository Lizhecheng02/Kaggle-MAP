# Map-Charting Student Math Misunderstandings

## 概要
このプロジェクトは、学生の数学の誤解を分類するための機械学習モデルを実装しています。学生の解答と説明から、どのような誤解（Misconception）を持っているかを予測します。

## fork-improvements.py について

### 機能
- 学生の数学問題への回答と説明から、誤解のカテゴリーと具体的な誤解を予測
- TF-IDFベクトル化による特徴抽出
- 10-fold交差検証による堅牢なモデル学習
- MAP@3（Mean Average Precision at 3）による評価

### 処理フロー
1. **データ前処理**
   - 質問文、回答、学生の説明を結合
   - テキストクリーニング（改行・記号除去、小文字化）
   - WordNetLemmatizerによる単語の基本形変換

2. **特徴抽出**
   - カテゴリー予測用：1-4gramのTF-IDF
   - 誤解予測用：1-3gramのTF-IDF
   - cumlライブラリによるGPU高速化

3. **モデル学習**
   - ロジスティック回帰を使用
   - カテゴリーと誤解を別々に予測
   - 10分割交差検証で汎化性能を確保

4. **予測と提出**
   - 各サンプルに対してトップ3の予測を生成
   - カテゴリー:誤解の形式で出力

### 依存ライブラリ
- pandas, numpy: データ処理
- cudf, cuml: GPU高速化されたデータ処理と機械学習
- scikit-learn: 評価指標
- nltk: 自然言語処理（WordNetLemmatizer）
- xgboost: 機械学習（インポートされているが未使用）

### 実行方法
```bash
python fork-improvements.py
```

### 入力データ
- `/kaggle/input/map-charting-student-math-misunderstandings/train.csv`: 訓練データ
- `/kaggle/input/map-charting-student-math-misunderstandings/test.csv`: テストデータ
- `/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv`: 提出フォーマット

### 出力
- `submission.csv`: Kaggle提出用ファイル（予測結果）

### 評価指標
- 精度（Accuracy）
- F1スコア（重み付き平均）
- MAP@3（Mean Average Precision at 3）

### 注意事項
- GPUメモリを大量に使用するため、CUDA対応のGPU環境が必要
- 初回実行時はNLTKのWordNetデータが自動的にダウンロードされます

## ファイルの違いと使い分け

### 1. fork-improvements.py（オリジナル版）
- **特徴**：
  - CuMLライブラリ（GPU）を使用
  - CategoryとMisconceptionを別々に予測する2段階アプローチ
  - LogisticRegressionモデルを使用
  - 10-fold交差検証
  - シンプルなテキストクリーニング（記号削除、小文字化）
  - TF-IDF特徴量のみ使用

### 2. fork-improvements-enhanced.py（拡張版）
- **特徴**：
  - LightGBMモデルを採用（CPU使用、より高精度）
  - 統合ターゲット（Category:Misconception）を直接予測
  - 高度な特徴量エンジニアリング：
    - 数学的特徴（分数、数値、演算子のカウント）
    - テキスト長と比率の特徴
    - 文字レベルのTF-IDF特徴も追加
  - QuestionIdごとの統計特徴を追加
  - 5-fold交差検証
  - より多くの特徴量（単語TF-IDF + 文字TF-IDF + 数値特徴）
  - **GPU使用**：TF-IDF計算時のみ（CuML）、その後CPUに転送

### 3. fork-improvements-optimized.py（最適化版）
- **特徴**：
  - XGBoostモデルを採用（CPU使用、バランスの良い性能）
  - enhanced版と同様の統合ターゲット予測
  - 計算効率を重視した設計：
    - TF-IDF特徴量を5000に制限（enhanced版は15000）
    - 3-fold交差検証で高速化
    - tree_method='hist'で学習高速化
  - scikit-learnのTfidfVectorizerを使用（互換性向上）
  - 実行時間とメモリ使用量を削減

### 4. fork-improvements-gpu.py（GPU最適化版）
- **特徴**：
  - XGBoostモデル（GPU版）を採用
  - CuMLでTF-IDF特徴量をGPU上で計算
  - tree_method='gpu_hist'でGPU高速化
  - 統合ターゲット予測
  - 3-fold交差検証
  - **GPU使用**：TF-IDF計算とXGBoost学習の両方でGPUを活用

## 各バージョンの使い分け

- **fork-improvements.py**：ベースラインとして、GPU環境でシンプルで高速な予測が必要な場合
- **fork-improvements-enhanced.py**：CPU環境で最高精度を求める場合（LightGBM使用）
- **fork-improvements-optimized.py**：CPU環境で精度と実行時間のバランスを重視する場合
- **fork-improvements-gpu.py**：GPU環境で最高のパフォーマンスを求める場合（推奨）

## 実装した改善点

### 1. データ分析による洞察
- Categoryの分布が不均衡（True_Correctが40%）
- Misconceptionの73%がNA（誤解なし）
- 65種類のターゲットクラスが存在

### 2. 特徴量エンジニアリングの改善
- **数学的特徴の抽出**：分数、数値、演算子の数をカウント
- **テキスト長特徴**：質問、回答、説明文の長さと比率
- **高度なテキストクリーニング**：数式パターンの検出と保護
- **Question別統計特徴**：質問ごとの回答傾向

### 3. モデルアーキテクチャの改善
- **元の実装**：CategoryとMisconceptionを独立に予測
- **改善版**：統合ターゲット（Category:Misconception）を直接予測
- **XGBoost/LightGBM採用**：より高精度な勾配ブースティング

### 4. その他の最適化
- 3-fold CVで高速化（元は10-fold）
- TF-IDF特徴量を5000に削減（メモリ効率化）
- tree_method='hist'で学習高速化

## 実行方法

```bash
# 仮想環境をアクティベート
source /root/kaggle/myenv/bin/activate

# データ分析を実行
python data_analysis.py

# 元の実装を実行
python fork-improvements.py

# 拡張版を実行（CPU環境、最高精度）
python fork-improvements-enhanced.py

# 最適化版を実行（CPU環境）
python fork-improvements-optimized.py

# GPU最適化版を実行（GPU環境、推奨）
python fork-improvements-gpu.py

nohup python script.py > script.log 2>&1 &
```

## 期待される改善効果
- MAP@3スコアの向上（統合予測による精度向上）
- より豊富な特徴量による予測精度の向上
- XGBoost/LightGBMによる非線形パターンの学習

## パフォーマンス比較（推定）
| バージョン | MAP@3スコア | 実行時間 | メモリ使用量 | GPU使用 |
|-----------|------------|----------|-------------|---------|
| オリジナル | ベースライン | 速い | 少ない | TF-IDF + モデル |
| 拡張版 | 最高 | 遅い | 多い | TF-IDFのみ |
| 最適化版 | 高い | 中程度 | 中程度 | なし |
| GPU最適化版 | 高い | 最速 | 中程度 | TF-IDF + モデル |
