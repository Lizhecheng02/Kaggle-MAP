#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
import sys
from typing import Dict, List, Tuple, Optional
import pandas as pd


def read_fold_best(results_file: Path,
                   metric_col: str = "eval/map@3",
                   step_col: str = "Step") -> Optional[Tuple[float, int]]:
    """
    读取单个 fold 的 results.txt，返回 (best_metric, best_step)。
    若文件不存在或格式不对，返回 None。
    """
    if not results_file.is_file():
        return None
    try:
        df = pd.read_csv(results_file, sep="\t")
        if metric_col not in df.columns or step_col not in df.columns:
            return None
        idx = df[metric_col].idxmax()
        row = df.loc[idx]
        return float(row[metric_col]), int(row[step_col])
    except Exception:
        return None


def version_number(name: str) -> Optional[int]:
    """从 'ver_23' 中提取 23；失败返回 None。"""
    m = re.fullmatch(r"ver_(\d+)", name)
    return int(m.group(1)) if m else None


def fold_number(name: str) -> Optional[int]:
    """从 'fold_4' 中提取 4；失败返回 None。"""
    m = re.fullmatch(r"fold_(\d+)", name)
    return int(m.group(1)) if m else None


def collect_version_stats(root: Path,
                          ver_start: int,
                          ver_end: int,
                          metric_col: str,
                          results_filename: str):
    """
    返回：
      - per_ver_details: Dict[ver_name, List[Dict]]，每个 ver 下每个 fold 的最佳 {fold, best_step, best_metric}
      - summary_rows: List[Dict]，每个 ver 的 {Version, Folds, Avg Best map@3}
    """
    per_ver_details: Dict[str, List[Dict]] = {}
    summary_rows: List[Dict] = []

    versions = sorted(
        [p for p in root.glob("ver_*") if p.is_dir()],
        key=lambda p: (version_number(p.name) is None, version_number(p.name) or -1)
    )

    for v in versions:
        vnum = version_number(v.name)
        if vnum is None or vnum < ver_start or vnum > ver_end:
            continue

        fold_rows: List[Dict] = []
        for fold_dir in sorted([d for d in v.glob("fold_*") if d.is_dir()],
                               key=lambda d: (fold_number(d.name) is None, fold_number(d.name) or -1)):
            res = read_fold_best(fold_dir / results_filename, metric_col=metric_col, step_col="Step")
            if res is None:
                continue
            best_metric, best_step = res
            fold_rows.append({
                "fold": fold_dir.name,
                "best_step": best_step,
                "best_metric": best_metric,
            })

        if fold_rows:
            per_ver_details[v.name] = fold_rows
            avg_score = sum(r["best_metric"] for r in fold_rows) / len(fold_rows)
            summary_rows.append({
                "Version": v.name,
                "Folds": len(fold_rows),
                "Avg Best map@3": round(avg_score, 4),
                "_avg_raw": avg_score,   # 内部排序用
                "_vnum": vnum            # 内部 tie-break 用
            })

    return per_ver_details, summary_rows


def main(root_dir: str,
         ver_start: int,
         ver_end: int,
         metric_col: str = "eval/map@3",
         results_filename: str = "results.txt"):

    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        print(f"根目录不存在：{root}", file=sys.stderr)
        sys.exit(1)

    per_ver_details, summary_rows = collect_version_stats(
        root, ver_start, ver_end, metric_col, results_filename
    )
    if not summary_rows:
        print("未找到任何可用结果。", file=sys.stderr)
        sys.exit(2)

    # 1) 打印各版本平均分排序表
    df = pd.DataFrame(summary_rows).drop(columns=["_avg_raw", "_vnum"])
    df_sorted = df.sort_values("Avg Best map@3", ascending=False)

    print("=== 各版本（ver_start..ver_end）每个 fold 最高分的平均值（降序） ===")
    try:
        print(df_sorted.to_markdown(index=False))
    except Exception:
        print(df_sorted.to_string(index=False))

    # 2) 找到平均分最高的版本（若并列，选版本号较大的）
    # 先按原始平均分降序，再按版本号升序（或降序都可，这里选降序，取第一个）
    top = sorted(summary_rows, key=lambda r: (r["_avg_raw"], r["_vnum"]), reverse=True)[0]
    top_ver = top["Version"]

    # 3) 列出该版本中每个 fold 的最佳 checkpoint（step 与分数）
    details = sorted(per_ver_details[top_ver],
                     key=lambda r: (fold_number(r["fold"]) is None, fold_number(r["fold"]) or -1))

    print(f"\n=== 平均值最高的版本：{top_ver}，对应每个 fold 的最佳 checkpoint ===")
    out_rows = []
    for r in details:
        out_rows.append({
            "Fold": r["fold"],
            "Best Step": r["best_step"],
            "Best map@3": round(r["best_metric"], 4),
        })
    df_details = pd.DataFrame(out_rows)
    try:
        print(df_details.to_markdown(index=False))
    except Exception:
        print(df_details.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="统计各版本下每个 fold 最高分并计算平均值；输出整体排序，并列出平均值最高版本的各 fold 最佳 checkpoint。"
    )
    parser.add_argument("--root", type=str, default=".", help="包含 ver_* 目录的根路径")
    parser.add_argument("--ver_start", type=int, default=52, help="起始版本号（默认 1）")
    parser.add_argument("--ver_end", type=int, default=59, help="结束版本号（默认 9999）")
    parser.add_argument("--metric", type=str, default="eval/map@3", help="指标列名（默认 eval/map@3）")
    parser.add_argument("--results", type=str, default="results.txt", help="每个 fold 下的结果文件名（默认 results.txt）")
    args = parser.parse_args()

    main(
        root_dir=args.root,
        ver_start=args.ver_start,
        ver_end=args.ver_end,
        metric_col=args.metric,
        results_filename=args.results,
    )
