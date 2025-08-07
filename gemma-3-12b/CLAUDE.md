# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

Kaggleコンペティション「Map Charting Student Math Misunderstandings」用のGemma-3-12Bモデルファインチューニングプロジェクト。学生の数学における誤解（misconceptions）を分類するタスク。

## 開発コマンド

### 学習の実行
```bash
python train.py
```

### 推論・提出ファイル作成
```bash
python submit.py
```

### デバッグ・検証用
```bash
python check_tokenizer.py    # トークナイザー設定の確認
python debug_dataset.py       # データセット処理の検証
```

## アーキテクチャ概要

### モデル構成
- **ベースモデル**: Gemma-3-12B (`/kaggle/input/gemma-3-12b`)
- **ファインチューニング手法**: LoRA (Low-Rank Adaptation)
- **分類タスク**: 36種類の数学的誤概念（misconceptions）への分類

### 主要コンポーネント

1. **カスタムモデルクラス** (`GemmaForSequenceClassification`)
   - Gemma-3の言語モデルを分類タスク用にカスタマイズ
   - 最終隠れ層の最後のトークンを使用して分類
   - LoRAアダプターによる効率的な学習

2. **データ処理フロー**
   - 入力: 問題文、学生の回答、正解/不正解フラグ、学生の説明
   - プロンプト形式でGemma-3用にフォーマット
   - 選択肢となる誤概念リストを含むプロンプト生成

3. **評価指標**
   - MAP@3 (Mean Average Precision at 3) - コンペティション公式指標
   - カスタムコールバックで最高スコアのモデルを自動保存

### 設定管理

`config.py`で全ての設定を一元管理:
- モデルパラメータ（エポック数、バッチサイズ、学習率など）
- LoRA設定（ランク、対象モジュール、ドロップアウト率）
- ファイルパス（データ、出力ディレクトリ）
- WandB設定（実験トラッキング）
- Early Stopping設定

### ディレクトリ構造

- `ver_{VER}/`: 学習成果物の保存先
  - `best/`: 最終的な最良モデル（LoRAアダプター）
  - `best_map3/`: MAP@3スコアが最高のモデル
  - `checkpoint-*/`: 学習途中のチェックポイント
  - `label_encoder.joblib`: ラベルエンコーダー

## 重要な実装詳細

- **GPU最適化**: 複数GPU対応、自動デバイス配置（device_map="auto"）
- **メモリ効率**: gradient accumulation、半精度演算の使用
- **動的パディング**: DataCollatorWithPaddingによるバッチ処理時の効率化
- **エラー対策**: NaN回避のための重み初期化、全精度での一部処理