# XGBoost Optuna最適化

Optunaを使用したハイパーパラメータ最適化統合版スクリプトです。

## ファイル構成

- `optimized_optuna.py`: Optuna統合版（最適化 + 訓練）
- `optimized_improve3.py`: ベースラインモデル（最適化前）

## 使用方法

```bash
# Optuna最適化を含む完全な実行
python optimized_optuna.py
```

## 設定

`optimized_optuna.py`内の設定:

```python
OPTUNA_CONFIG = {
    'n_trials': 50,         # 試行回数
    'n_folds_optuna': 3,    # 最適化時のfold数（高速化）
    'n_folds_final': 5,     # 最終訓練時のfold数
    'optimize': True,       # False: 保存済みパラメータを使用
}
```

## 最適化対象パラメータ

### XGBoostパラメータ
- `max_depth`: 8-20
- `learning_rate`: 0.01-0.3 (log scale)
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.6-1.0
- `min_child_weight`: 1-10
- `gamma`: 0.0-0.5
- `alpha`: 0.0-1.0
- `lambda`: 0.5-3.0
- `max_bin`: [128, 256, 512]
- `max_delta_step`: 0-5

### TF-IDFパラメータ
- `ngram_max`: 2-4
- `max_df`: 0.8-0.95
- `min_df`: 1-5
- `max_features`: 3000-10000

## 特徴

- **GPU最適化**: `tree_method='gpu_hist'`を使用
- **早期終了**: MedianPrunerによる効率的な探索
- **並列処理**: テキスト前処理の高速化
- **統合設計**: 1つのファイルで最適化から訓練まで完結

## 出力ファイル

- `best_params_optuna.json`: 最適化されたパラメータ
- `submission_optuna.csv`: 提出用ファイル

## 実行フロー

1. データ読み込み（cuDF使用）
2. テキスト前処理（並列処理）
3. 特徴量抽出（GPU活用）
4. Optuna最適化（50試行、3-fold CV）
5. 最適パラメータで最終訓練（5-fold CV）
6. 提出ファイル生成
