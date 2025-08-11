# 精度向上のための追加Tips

## 実装済みの改善点

1. **データ拡張** (`data_augmentation.py`)
   - 少数クラスのオーバーサンプリング
   - パラフレーズによる拡張
   - 合成データの生成

2. **マルチタスク学習** (`multitask_model.py`)
   - 正解/不正解の判定
   - カテゴリ分類（Correct/Misconception/Neither）
   - 具体的なMisconception分類
   - タスク間の情報共有による性能向上

3. **最適化されたハイパーパラメータ** (`config_v2.py`)
   - より大きなモデル（DeBERTa-v3-base）
   - 最適な学習率とスケジューラー
   - Focal Lossによるクラス不均衡対策
   - ラベルスムージング

4. **推論時の工夫** (`inference_v2.py`)
   - Test Time Augmentation (TTA)
   - 信頼度に基づく予測調整

## さらなる改善のための提案

### 1. 特徴量エンジニアリングの強化
```python
# 追加の特徴量例
- 説明文の長さ
- 数式の有無と数
- 特定のキーワードの出現（"because", "therefore"など）
- 問題文と回答の類似度
```

### 2. 外部データの活用
- 数学教育関連のテキストデータでの事前学習
- 類似の教育データセットからの転移学習

### 3. エラー分析
```python
# validation setでの誤分類サンプルの分析
def analyze_errors(predictions, true_labels, test_data):
    errors = predictions != true_labels
    error_samples = test_data[errors]
    # Misconceptionごとのエラー率を分析
    # 混同行列の作成
    # 難しいサンプルの特徴分析
```

### 4. 後処理の最適化
```python
# ルールベースの後処理
def post_process_predictions(predictions, test_data):
    # 矛盾する予測の修正
    # （例：True_Correctなのに具体的なMisconceptionが予測される）
    # QuestionIdごとの一貫性チェック
```

### 5. 最適なしきい値の調整
```python
# Validationデータでの最適しきい値の探索
def find_optimal_thresholds(val_predictions, val_labels):
    # 各クラスごとの最適しきい値
    # MAP@3を最大化するしきい値の探索
```

## トレーニング実行例

```bash
# 基本的なトレーニング
python train_v2.py

# カスタム設定でのトレーニング
python train_v2.py --epochs 40 --learning_rate 1e-5

# 推論の実行
python inference_v2.py
```

## デバッグとモニタリング

1. **トレーニング中のモニタリング**
   - 各エポックでのMAP@3スコア
   - 学習率の推移
   - 損失関数の値（メインタスクと補助タスク）

2. **予測の可視化**
   - 予測確率の分布
   - クラスごとの性能
   - 混同行列

## 注意事項

- GPUメモリが不足する場合は、`TRAIN_BATCH_SIZE`を調整
- 大きなモデルを使用する場合は、`gradient_accumulation_steps`を増やす
- 過学習の兆候が見られる場合は、`dropout`率を上げる