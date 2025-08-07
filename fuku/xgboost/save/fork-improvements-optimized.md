# 日本語の説明

## スクリプト名
`fork-improvements-optimized.py`

## 概要
このスクリプトは、Kaggleの「Map-Charting Student Math Misunderstandings」コンペティションのために作成された、モデルの学習と予測を最適化したバージョンです。GPU（NVIDIA CUDA）を利用して処理を高速化し、特徴量エンジニアリングとハイパーパラメータ調整により、高い精度を目指します。

## 主な機能
1.  **データの前処理**:
    *   `train.csv`と`test.csv`を読み込みます。
    *   テキストデータを結合し、数式（分数など）を特別なトークンに置き換える高度なクリーニングを行います。
    *   NLTKライブラリを使用してテキストをレンマ化（見出し語化）します。

2.  **特徴量エンジニアリング**:
    *   質問、回答、説明文の長さやその比率を特徴量として追加します。
    *   テキストに含まれる分数、数値、演算子の数を抽出し、特徴量として加えます。

3.  **TF-IDFベクトル化**:
    *   クリーニングされたテキストから、TF-IDF（Term Frequency-Inverse Document Frequency）特徴量を生成します。語彙数を5000に制限し、計算効率を向上させています。

4.  **モデル学習**:
    *   GPUで高速に動作するXGBoost（`tree_method='gpu_hist'`）を使用します。
    *   層化K分割交差検証（`StratifiedKFold`）を用いて、モデルの汎化性能を高めます。
    *   学習率、木の深さ、正則化項などのハイパーパラメータが調整されています。

5.  **評価と予測**:
    *   学習済みモデルを使い、検証データとテストデータの予測を行います。
    *   コンペティションの評価指標であるMAP@3（Mean Average Precision at 3）を計算して、モデルの性能を評価します。
    *   最終的な予測結果を`submission.csv`として出力します。

## 実行方法
このスクリプトは、Kaggle環境などのGPUが利用可能な環境で実行することを想定しています。必要なライブラリ（pandas, cudf, cuml, scikit-learn, nltk, xgboost）をインストールした後、直接実行してください。

---

# English Description

## Script Name
`fork-improvements-optimized.py`

## Overview
This script is an optimized version for model training and prediction, created for the Kaggle competition "Map-Charting Student Math Misunderstandings." It leverages a GPU (NVIDIA CUDA) to accelerate processing and aims for high accuracy through advanced feature engineering and hyperparameter tuning.

## Key Features
1.  **Data Preprocessing**:
    *   Reads `train.csv` and `test.csv`.
    *   Combines text data and performs advanced cleaning, such as replacing mathematical expressions (e.g., fractions) with special tokens.
    *   Lemmatizes the text using the NLTK library.

2.  **Feature Engineering**:
    *   Adds features based on the length of questions, answers, explanations, and their ratios.
    *   Extracts the counts of fractions, numbers, and operators from the text to use as features.

3.  **TF-IDF Vectorization**:
    *   Generates TF-IDF (Term Frequency-Inverse Document Frequency) features from the cleaned text. The vocabulary size is limited to 5000 to improve computational efficiency.

4.  **Model Training**:
    *   Uses XGBoost with GPU acceleration (`tree_method='gpu_hist'`).
    *   Employs `StratifiedKFold` cross-validation to enhance the model's generalization performance.
    *   Hyperparameters such as learning rate, tree depth, and regularization terms have been tuned.

5.  **Evaluation and Prediction**:
    *   Uses the trained model to make predictions on validation and test data.
    *   Evaluates the model's performance by calculating the MAP@3 (Mean Average Precision at 3) score, which is the competition's evaluation metric.
    *   Outputs the final predictions to `submission.csv`.

## How to Run
This script is intended to be run in an environment with a GPU, such as a Kaggle Notebook. After installing the required libraries (pandas, cudf, cuml, scikit-learn, nltk, xgboost), execute the script directly.
