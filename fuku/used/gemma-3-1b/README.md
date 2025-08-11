
# Qwen3-0.6Bモデル ファインチューニング プロジェクト

このプロジェクトは、Kaggleコンペティション「[Map Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings)」のために、Qwen3-0.6Bモデルをファインチューニングし、学生の数学における誤解を分類することを目的としています。

## ディレクトリ構造

```
.
├── check_tokenizer.py
├── config_sample.py
├── config.py
├── data_collator.py
├── debug_dataset.py
├── README.md
├── submit.py
├── train.log
├── train.py
├── utils.py
├── __pycache__/
└── ver_2/
    ├── checkpoint-1200/
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   └── ... (その他モデルファイル)
    └── token_length_distribution.png
```

## ファイルの役割

| ファイル名 | 役割 |
| :--- | :--- |
| `train.py` | メインの学習スクリプト。データの読み込み、前処理、モデルのファインチューニング、評価、モデルの保存を行います。 |
| `submit.py` | 学習済みモデルを使用して、テストデータに対する予測を生成し、提出用の`submission.csv`ファイルを作成します。 |
| `config.py` | モデル名、ハイパーパラメータ、ファイルパスなど、プロジェクト全体の設定を管理します。 |
| `utils.py` | データの前処理、入力テキストのフォーマット、評価指標の計算など、複数のスクリプトで共有されるヘルパー関数を格納します。 |
| `data_collator.py` | `transformers.Trainer`がバッチ処理を行う際に、動的にパディングを行うためのカスタムデータコレーターを定義します。 |
| `check_tokenizer.py` | 使用するトークナイザーの語彙数やパディング設定などを確認・検証するためのスクリプトです。 |
| `debug_dataset.py` | データセットのトークナイズ処理やパディングが正しく行われているかを確認するためのデバッグ用スクリプトです。 |
| `ver_2/` | 学習の成果物（チェックポイント、ログ、グラフなど）を保存するディレクトリ。バージョン管理されています。 |
| `ver_2/checkpoint-1200/` | 学習のチェックポイント。LoRAアダプターの重み(`adapter_model.safetensors`)などが含まれます。 |
| `ver_2/token_length_distribution.png` | 入力テキストのトークン長の分布を可視化したグラフ。`MAX_LEN`設定の参考にします。 |

## 実行フロー

1.  **設定**:
    *   `config.py`で使用するモデル（`MODEL_NAME`）、エポック数（`EPOCHS`）、バッチサイズ（`TRAIN_BATCH_SIZE`）などのハイパーパラメータを設定します。

2.  **学習の実行**:
    *   `train.py`を実行して、モデルのファインチューニングを開始します。
    *   スクリプトは`config.py`から設定を読み込み、`train.csv`を処理します。
    *   学習済みモデル（LoRAアダプター）とラベルエンコーダーは`ver_2/`ディレクトリに保存されます。

    ```bash
    python train.py
    ```

3.  **推論と提出ファイルの作成**:
    *   学習が完了したら、`submit.py`を実行します。
    *   スクリプトは`ver_2/`に保存されたモデルとラベルエンコーダーを読み込み、`test.csv`データに対する予測を行います。
    *   最終的な提出ファイル`submission.csv`がプロジェクトのルートディレクトリに生成されます。

    ```bash
    python submit.py
    ```

## モデルと学習戦略

*   **ベースモデル**: `Qwen-3-0.6b` を使用しています。
*   **ファインチューニング手法**:
    *   **LoRA (Low-Rank Adaptation)** を採用し、計算効率の良いファインチューニングを実現しています。`train.py`内で`PeftModel`を通じてLoRAアダプターをモデルに適用しています。
    *   これにより、元のモデルの大部分の重みを固定したまま、少数のパラメータ（アダプター）のみを更新するため、メモリ消費を抑えつつ高速な学習が可能です。
*   **入力形式**:
    *   `utils.py`の`format_input`関数で定義されているように、問題文、学生の回答、その回答が正解か不正解か、そして学生の説明を組み合わせたプロンプトをモデルへの入力としています。
*   **評価指標**:
    *   コンペティションの評価指標である `MAP@3` (Mean Average Precision at 3) を使用して、モデルの性能を評価します。

