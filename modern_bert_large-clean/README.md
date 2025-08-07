# Deberta Student Math Misunderstandings Classifier

このプロジェクトは、Kaggleの「MAP - Charting Student Math Misunderstandings」コンペティション用のDebertaモデル実装です。学生の数学問題への回答と説明から、誤概念を予測します。

## プロジェクト構造

```
deberta_xsmall/
├── config.py       # 設定ファイル（パス、パラメータなど）
├── utils.py        # 共通ユーティリティ関数
├── train.py        # モデルトレーニング用スクリプト
├── submit.py       # 推論・提出ファイル生成用スクリプト
└── README.md       # このファイル
```

## 必要なライブラリ

```bash
pip install transformers datasets pandas numpy scikit-learn torch matplotlib joblib
```

## 使用方法

### 1. モデルのトレーニング

```bash
python train.py
```

このコマンドを実行すると：
- 訓練データの読み込みと前処理
- 特徴量エンジニアリング（正解/不正解の判定）
- モデルのファインチューニング
- 最良モデルとラベルエンコーダーの保存

出力ファイル：
- `ver_1/best/` - トレーニング済みモデル
- `ver_1/label_encoder.joblib` - ラベルエンコーダー
- `ver_1/token_length_distribution.png` - トークン長の分布図

### 2. 推論と提出ファイルの生成

```bash
python submit.py
```

config.pyで定義された以下の設定が自動的に使用されます：
- モデルパス: `ver_1/best`
- エンコーダーパス: `ver_1/label_encoder.joblib`
- テストデータパス: `/kaggle/input/map-charting-student-math-misunderstandings/test.csv`
- 訓練データパス: `/kaggle/input/map-charting-student-math-misunderstandings/train.csv`
- 出力ファイル: `submission.csv`

**注意**: 設定を変更する場合は、config.pyファイルを直接編集してください。

## 設定の変更

`config.py`ファイルで以下の設定を変更できます：

- `VER`: バージョン番号（出力ディレクトリ名に使用）
- `MODEL_NAME`: 使用するDebertaモデル
- `EPOCHS`: エポック数
- `MAX_LEN`: 最大トークン長
- `TRAIN_BATCH_SIZE`: 訓練時のバッチサイズ
- `EVAL_BATCH_SIZE`: 評価時のバッチサイズ
- `LEARNING_RATE`: 学習率
- `SUBMISSION_OUTPUT_PATH`: 提出ファイルのデフォルト出力パス
- その他のトレーニングパラメータ

## 特徴

1. **データ前処理**
   - 正解/不正解の判定を特徴量として追加
   - 質問文、回答、正誤状態、学生の説明を組み合わせたプロンプト生成

2. **評価指標**
   - MAP@3（Mean Average Precision at 3）を使用
   - Top-3の予測精度を評価

3. **モジュール化**
   - 設定、ユーティリティ、トレーニング、推論を分離
   - 再利用可能なコード構造

## トラブルシューティング

- メモリ不足の場合：`config.py`でバッチサイズを小さくしてください
- トークン長オーバーの場合：`MAX_LEN`を調整してください
- GPU使用の設定：環境変数`CUDA_VISIBLE_DEVICES`で制御可能

## 注意事項

- データパスはKaggle環境用に設定されています。ローカル環境で使用する場合は`config.py`のパスを適切に変更してください
- モデルファイルは大きいため、十分なディスク容量を確保してください