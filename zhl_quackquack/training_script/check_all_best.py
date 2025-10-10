#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import argparse
import re
import sys
from typing import Dict, Tuple, Optional, List
from pathlib import Path


def read_best_of_one_results(results_file: Path, metric_col: str = "eval/map@3", step_col: str = "Step") -> Optional[Tuple[int, float]]:
    """
    从单个 results.txt 中读取，并返回 metric 最大值对应的 (step, metric)。
    若文件不存在或列缺失，则返回 None。
    """
    if not results_file.is_file():
        return None
    try:
        # 文件为制表符分隔
        df = pd.read_csv(results_file, sep="\t")
        if metric_col not in df.columns or step_col not in df.columns:
            return None
        # 找到 metric 最大的行
        idx = df[metric_col].idxmax()
        row = df.loc[idx]
        return int(row[step_col]), float(row[metric_col])
    except Exception:
        return None


def version_number(name: str) -> Optional[int]:
    """
    从 'ver_23' 提取整数 23；若匹配不上返回 None。
    """
    m = re.fullmatch(r"ver_(\d+)", name)
    if not m:
        return None
    return int(m.group(1))


def find_folds(base: Path) -> List[str]:
    """
    发现所有 fold 目录名（如 fold_0 ...），按数字序排序返回名字列表。
    """
    folds = []
    for p in base.glob("ver_*"):
        if p.is_dir():
            for f in p.glob("fold_*"):
                if f.is_dir():
                    folds.append(f.name)
    # 去重并按 fold 编号排序
    unique = {}
    for name in folds:
        m = re.fullmatch(r"fold_(\d+)", name)
        if m:
            unique[int(m.group(1))] = name
    return [unique[k] for k in sorted(unique.keys())]


def main(
    root_dir: str,
    ver_start: int,
    ver_end: int,
    metric_col: str = "eval/map@3",
    results_filename: str = "results.txt",
):
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        print(f"根目录不存在：{root}", file=sys.stderr)
        sys.exit(1)

    # 发现 fold 列表（自动识别，不强依赖固定 0..4）
    folds = find_folds(root)
    if not folds:
        print("未在任何 ver_* 目录下发现 fold_* 目录。", file=sys.stderr)
        sys.exit(1)

    # 初始化每个 fold 的最佳记录：fold -> (best_ver_name, best_step, best_metric)
    best: Dict[str, Tuple[str, int, float]] = {}

    # 遍历版本范围（含端点）
    for v in sorted([p for p in root.glob("ver_*") if p.is_dir()],
                    key=lambda p: version_number(p.name) if version_number(p.name) is not None else -1):
        vnum = version_number(v.name)
        if vnum is None or vnum < ver_start or vnum > ver_end:
            continue

        for fold_name in folds:
            fold_dir = v / fold_name
            if not fold_dir.is_dir():
                continue
            results_file = fold_dir / results_filename
            best_pair = read_best_of_one_results(results_file, metric_col=metric_col, step_col="Step")
            if best_pair is None:
                continue
            step, metric = best_pair

            if fold_name not in best or metric > best[fold_name][2]:
                best[fold_name] = (v.name, step, metric)

    if not best:
        print("在指定版本范围内未找到任何可用的 results.txt 或目标指标。", file=sys.stderr)
        sys.exit(2)

    # 组装输出表
    rows = []
    for fold_name in sorted(best.keys(), key=lambda n: int(re.search(r"\d+", n).group())):
        vname, step, metric = best[fold_name]
        rows.append({
            "Fold": fold_name,
            "Best Version": vname,
            "Best Step": step,
            "Best map@3": round(metric, 4),
        })

    df = pd.DataFrame(rows)
    # 为了和示例相似（可按需要改排序规则）
    # 这里按 Fold 升序；若想按分数降序：df.sort_values("Best map@3", ascending=False, inplace=True)
    # df.sort_values("Best map@3", ascending=False, inplace=True)

    print("=== 每个 fold 的最终最佳版本 ===")
    # 需要 markdown 风格的表格
    try:
        print(df.to_markdown(index=False))
    except Exception:
        # 某些环境无 tabulate 依赖时，退化为普通打印
        print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计指定版本范围内各 fold 的最佳 eval/map@3。")
    parser.add_argument("--root", type=str, default=".", help="包含 ver_* 目录的根路径")
    parser.add_argument("--ver_start", type=int, default=52, help="起始版本号（默认 1，表示 ver_1）")
    parser.add_argument("--ver_end", type=int, default=59, help="结束版本号（默认 9999，几乎包含所有版本）")
    parser.add_argument("--metric", type=str, default="eval/map@3", help="指标列名（默认 eval/map@3）")
    parser.add_argument("--results", type=str, default="results.txt", help="每个 fold 下结果文件名（默认 results.txt）")
    args = parser.parse_args()

    main(
        root_dir=args.root,
        ver_start=args.ver_start,
        ver_end=args.ver_end,
        metric_col=args.metric,
        results_filename=args.results,
    )
