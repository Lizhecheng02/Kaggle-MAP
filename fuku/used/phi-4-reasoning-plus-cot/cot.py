# read_parquet_basic.py
# 使い方:
#   pip install pandas pyarrow   # or: pip install pandas fastparquet
#   python read_parquet_basic.py /path/to/file.parquet --head 5 --to-csv out.csv

import argparse
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="入力Parquetファイルのパス")
    p.add_argument("--head", type=int, default=5, help="先頭n行を表示（デフォルト5）")
    p.add_argument("--to-csv", default=None, help="CSVに書き出すパス（任意）")
    p.add_argument("--to-jsonl", default=None, help="JSONLに書き出すパス（任意）")
    args = p.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pandas が見つかりません。`pip install pandas` を実行してください。", file=sys.stderr)
        sys.exit(1)

    # エンジンは自動選択（pyarrow か fastparquet が必要）
    try:
        df = pd.read_parquet(args.path)
    except ImportError:
        print(
            "Parquetの読み込みに pyarrow または fastparquet が必要です。\n"
            "例: pip install pyarrow   または   pip install fastparquet",
            file=sys.stderr
        )
        sys.exit(1)

    # 基本情報
    print("=== Basic Info ===")
    print(f"shape: {df.shape}")
    print(f"columns ({len(df.columns)}): {list(df.columns)}")

    # 先頭n行を表示
    if args.head > 0:
        print("\n=== Head ===")
        print(df.head(args.head))

    # 変換オプション
    if args.to_csv:
        df.to_csv(args.to_csv, index=False)
        print(f"\n[Saved] CSV -> {args.to_csv}")

    if args.to_jsonl:
        df.to_json(args.to_jsonl, orient="records", lines=True, force_ascii=False)
        print(f"[Saved] JSONL -> {args.to_jsonl}")

if __name__ == "__main__":
    main()
