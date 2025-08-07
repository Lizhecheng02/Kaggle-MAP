"""
Validationデータを除いたTrainデータを出力するスクリプト
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from config import TRAIN_DATA_PATH, SYNTHETIC_DATA_PATH, USE_SYNTHETIC_DATA, VALIDATION_SPLIT, RANDOM_SEED

def main():
    """メイン関数"""
    
    # 元のtrainデータを読み込み
    print("Loading training data...")
    train = pd.read_csv(TRAIN_DATA_PATH)
    original_train_shape = train.shape
    
    # 合成データの読み込み（必要な場合）
    if USE_SYNTHETIC_DATA:
        print("Loading synthetic data...")
        synthetic_data = pd.read_csv(SYNTHETIC_DATA_PATH, encoding='utf-8', encoding_errors='replace')
        print(f"Synthetic data shape: {synthetic_data.shape}")
        
        # 元のデータと合成データを結合
        train = pd.concat([train, synthetic_data], ignore_index=True)
        print(f"Combined data shape: {train.shape} (original: {original_train_shape[0]}, synthetic: {synthetic_data.shape[0]})")
    
    # train.pyと同じ前処理を適用
    train['Misconception'] = train.Misconception.fillna('NA')
    train['target'] = train.Category + ":" + train.Misconception
    
    # train.pyと同じ分割ロジックを使用
    if USE_SYNTHETIC_DATA:
        # オリジナルデータのインデックスを記録
        original_indices = list(range(original_train_shape[0]))
        # オリジナルデータのみを使用して検証データを作成
        original_data = train.iloc[original_indices]
        train_original_df, val_df = train_test_split(original_data, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
        
        # 合成データをトレーニングデータに追加
        synthetic_indices = list(range(original_train_shape[0], len(train)))
        synthetic_df = train.iloc[synthetic_indices]
        train_df = pd.concat([train_original_df, synthetic_df], ignore_index=True)
        
        print(f"Training set: {len(train_df)} samples (original: {len(train_original_df)}, synthetic: {len(synthetic_df)})")
        print(f"Validation set: {len(val_df)} samples (original data only)")
    else:
        train_df, val_df = train_test_split(train, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
        print(f"Training set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")
    
    # validationデータを除いたtrainデータを保存
    output_path = 'exclude_valid.csv'
    train_df.to_csv(output_path, index=False)
    print(f"\nValidationデータを除いたTrainデータを {output_path} に保存しました。")
    print(f"保存されたデータ数: {len(train_df)} 行")
    
    # 検証用：元のデータとの比較
    print(f"\n検証:")
    print(f"元のデータ数: {len(train)}")
    print(f"Training数: {len(train_df)}")
    print(f"Validation数: {len(val_df)}")
    print(f"合計: {len(train_df) + len(val_df)}")

if __name__ == "__main__":
    main()