# 改善施策リスト

## 現在のベースライン
- モデル: ModernBERT-large
- CVスコア: 0.940 (MAP@3)
- エポック数: 3
- 最大トークン長: 256
- 学習率: 2e-5
- バッチサイズ: train=32, eval=64

## 試行済み
- [x] DeBERTa-v3-large（既に試行済み）

## 改善施策（優先度順）

### 1. 入力プロンプトの最適化 【試行済，精度下がる】
- 説明文を先頭に配置（recency biasを活用）
- 特殊トークン（<CORRECT>/<WRONG>）の追加
- フィールドマーカーの明確化
- 期待効果: +0.1-0.2 MAP
- 結果: 0.940 → 0.880

### 2. Multi-sample dropout head 【実装済み】
- 5つのdropoutヘッド（p=0.5）で予測の分散を削減
- 最終的にlogitsを平均化
- 期待効果: +0.1-0.2 MAP
精度向上0.938→0.940

### 3. 5-fold Cross Validation
- 現在のvalidation split 0.2を活用
- CategoryでStratified分割
- より堅牢な評価とアンサンブルの基盤
- 期待効果: より安定した性能

### 4. データ拡張
- バックトランスレーション（日本語→英語→日本語）
- 説明文のパラフレーズ生成
- Character-level noise（10%のランダムな文字swap/drop）
- 期待効果: +0.2-0.3 MAP

### 5. Adversarial Training (FGM)
- Fast Gradient Method実装
- ε=1e-2で勾配ベースの摂動を追加
- 期待効果: +0.1 MAP

### 6. Loss関数の改善
- Label smoothing（ε=0.05）
- Focal loss（γ=2）の追加
- Cross Entropy + Ranking lossの組み合わせ
- 期待効果: +0.1-0.2 MAP

### 7. kNN Retrieval（FAISS）
- CLS埋め込みでインデックス構築
- k=10-20の近傍検索
- p=0.7でモデル予測とブレンド
- 期待効果: +0.15 MAP

### 8. トレーニング戦略の改善
- Gradual unfreeze（段階的な層の解凍）
- SWA（Stochastic Weight Averaging）
- 長めのトレーニング（4-5エポック）
- 期待効果: +0.1-0.2 MAP

### 9. アンサンブル
- 複数シード（4-5シード）
- Rank averaging
- Dropout-TTA（20パス平均）
- 期待効果: +0.2-0.3 MAP

### 10. 追加の特徴量
- テキスト長、数学記号の数など
- Character-level CNN branch
- 期待効果: +0.05-0.1 MAP

## 総合期待効果
これらの施策を組み合わせることで、0.948-0.951のスコア達成を目指す。
