#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
submit.py — 学習なし後処理（masking/prior/keyword/true-false bias/辞書）で MAP@3 を底上げする提出生成スクリプト

使い方（例）
-----------
1) 予測（logits or 確率）を CSV で用意する:
   - 形式: 1列目に row_id、2列目以降に **ラベル名（"Category:Misconception"）** を列名とした数値
   - 例: predictions.csv

2) 実行:
   python submit.py \
       --train /mnt/data/train.csv \
       --test  /path/to/test.csv \
       --pred  /path/to/predictions.csv \
       --out   /path/to/submission.csv \
       --alpha 1.5 --beta 1.0 --gamma 3.0 --exact-threshold 0.9

   ※ "--pred" が .npy/.npz の場合は "--labels label_list.json" が必要（logits列の順序定義）。
   ※ 出力は Kaggle MAP@3 形式（"row_id,prediction" で prediction は空白区切りの上位3ラベル）。

主な処理
--------
- QuestionId ごとの Misconception 出現集合で "*_Misconception" ラベルをマスク（未出は無効化）
- QuestionId 事前分布 P(label | QuestionId) を log で加算（係数 alpha）
- MC_Answer がその問題の正答（学習CSVで最頻の True_* の答え）と一致 ⇒ True_* にバイアス、
  一致しない ⇒ False_* にバイアス（係数 gamma）
- 学習CSVから抽出した高精度キーワードで該当 Misconception を加点（係数 beta）
- 学習CSVの完全一致辞書（説明文→ラベル）が高一致率ならハード／ソフトに反映（--exact-threshold）

依存: pandas, numpy
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Set

import numpy as np
import pandas as pd


# ------------------------ ユーティリティ ------------------------

def logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    res = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if axis is not None:
        res = np.squeeze(res, axis=axis)
    return res


def to_logit_space(x: np.ndarray) -> np.ndarray:
    """
    入力が確率（各行で和=1）っぽければ log に変換。そうでなければそのまま（logits とみなす）。
    """
    # 0-1範囲かつ行和が ~1 の場合を確率とみなす
    if x.min() >= -1e-9 and x.max() <= 1 + 1e-9:
        row_sums = x.sum(axis=1)
        if np.all(np.isfinite(row_sums)) and np.allclose(row_sums, 1.0, atol=1e-3):
            x = np.clip(x, 1e-12, 1.0)
            return np.log(x)
    return x  # 既に logits と判断


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def normalize_text(t: str) -> str:
    t = (t or "").strip().lower()
    # 記号・重空白の簡易正規化
    t = re.sub(r"[\r\n\t]", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)  # 句読点→空白
    t = re.sub(r"\s+", " ", t).strip()
    return t


def split_label(label: str) -> Tuple[str, str]:
    """
    'Category:Misconception' を (Category, Misconception) に分解
    """
    if ":" not in label:
        raise ValueError(f"Unexpected label format (no colon): {label}")
    cat, mc = label.split(":", 1)
    return cat, mc


# ------------------------ キーワード辞書（高精度ワードのみ） ------------------------

KEYWORD_HINTS: Dict[str, str] = {
    # Misconception -> regex (case-insensitive)
    "Additive": r"\b(add|plus|together|sum|difference|between)\b",
    "Subtraction": r"\b(subtract|minus|take away|less)\b",
    "Mult": r"\b(multiply|times|groups of|product)\b",
    "Division": r"\b(divide|quotient|each|per)\b",
    "Inversion": r"\b(flip|reciprocal|keep change flip|invert|turn)\b",
    "Denominator-only_change": r"\b(denominator|bottom)\b(?!.*\bnumerator\b)",
    "Whole_numbers_larger": r"\b(whole number|larger|bigger)\b",
    "SwapDividend": r"\b(swap|switch)\b.*\bdivide\b.*\b(by|with)\b.*\b(top|numerator)\b",
    "Scale": r"\b(scale|ratio|proportional|proportion)\b",
    "Incomplete": r"\b(i don.?t know|idk|not sure)\b|\?$",
    "Irrelevant": r"\b(color|colour|shaded|blue|white|picture|triangle|rectangle)\b",
    "WNB": r"\b(shaded|not|blank|white|simplify)\b",
    "Adding_terms": r"\b(add.*term|combine.*term)\b",
    "Inverse_operation": r"\b(inverse operation)\b",
    "Wrong_fraction": r"\b(fraction|numerator|denominator|top|bottom)\b",
    "Wrong_Fraction": r"\b(fraction|numerator|denominator|top|bottom)\b",
    "Duplication": r"\b(double|twice|duplicate|count.*twice)\b",
    "Positive": r"\b(positive|negatives? (don.?t|do not) matter)\b",
}

KEYWORD_HINTS = {k: re.compile(v, re.I) for k, v in KEYWORD_HINTS.items()}


# ------------------------ メタ構造体 ------------------------

@dataclass
class Meta:
    labels: List[str]  # すべての 'Category:Misconception' ラベル（pred 列順）
    label_index: Dict[str, int]
    # QuestionId -> Allowed Misconception（*_Misconception ラベルにのみ適用）
    qid_allowed_mc: Dict[int, Set[str]]
    # QuestionId -> 事前分布（Laplace 平滑）: log P(label | QuestionId)
    qid_log_prior: Dict[int, np.ndarray]
    # QuestionId -> 正答（True_* の最頻 MC_Answer）
    qid_true_answer: Dict[int, str]
    # 完全一致辞書: norm(StudentExplanation) -> (label, agreement)
    exact_match_dict: Dict[str, Tuple[str, float]]


# ------------------------ メタ構築 ------------------------

def build_meta_from_train(train_csv: str, labels_in_pred: List[str],
                          exact_threshold: float = 0.9) -> Meta:
    df = pd.read_csv(train_csv)
    # 正規化
    if "Misconception" in df.columns:
        df["Misconception"] = df["Misconception"].fillna("NA").astype(str)
    else:
        raise ValueError("train.csv に Misconception 列が見つかりません。")
    if "Category" not in df.columns:
        raise ValueError("train.csv に Category 列が見つかりません。")
    if "QuestionId" not in df.columns:
        raise ValueError("train.csv に QuestionId 列が見つかりません。")
    if "MC_Answer" not in df.columns:
        raise ValueError("train.csv に MC_Answer 列が見つかりません。")
    if "StudentExplanation" not in df.columns:
        raise ValueError("train.csv に StudentExplanation 列が見つかりません。")

    df["target"] = df["Category"].astype(str) + ":" + df["Misconception"].astype(str)

    # ラベル順（pred の列と一致している必要）
    labels = list(labels_in_pred)
    label_index = {lbl: i for i, lbl in enumerate(labels)}

    # QuestionId -> 出現した Misconception（*_Misconception のみ）
    is_mis = df["Category"].astype(str).str.endswith("_Misconception")
    qid_allowed_mc: Dict[int, Set[str]] = defaultdict(set)
    for qid, sub in df.loc[is_mis, ["QuestionId", "Misconception"]].dropna().groupby("QuestionId"):
        qid_allowed_mc[int(qid)] = set(m for m in sub["Misconception"].astype(str).tolist())

    # QuestionId -> 事前分布（ラベル軸）
    # counts[label_idx] for each qid
    qid_label_counts: Dict[int, np.ndarray] = {}
    for qid, sub in df.groupby("QuestionId"):
        cnt = np.zeros(len(labels), dtype=np.float64)
        for t, c in Counter(sub["target"]).items():
            if t in label_index:
                cnt[label_index[t]] = c
        # Laplace 平滑
        s = 1.0
        cnt += s
        cnt /= cnt.sum()
        qid_label_counts[int(qid)] = np.log(cnt)

    # QuestionId -> 正答（True_* の最頻 MC_Answer）
    qid_true_answer: Dict[int, str] = {}
    is_true = df["Category"].astype(str).str.startswith("True_")
    for qid, sub in df.loc[is_true, ["QuestionId", "MC_Answer"]].dropna().groupby("QuestionId"):
        ans = sub["MC_Answer"].astype(str)
        if not ans.empty:
            mode = ans.value_counts().idxmax()
            qid_true_answer[int(qid)] = mode

    # 完全一致辞書（学習内の説明文 -> 最頻ラベル、合意率）
    norm_text = df["StudentExplanation"].astype(str).map(normalize_text)
    df_text = pd.DataFrame({"text": norm_text, "target": df["target"]})
    exact_match_dict: Dict[str, Tuple[str, float]] = {}
    for text, sub in df_text.groupby("text"):
        if not text:
            continue
        vc = sub["target"].value_counts()
        top_label = vc.index[0]
        agree = float(vc.iloc[0]) / float(vc.sum())
        if agree >= exact_threshold:
            exact_match_dict[text] = (top_label, agree)

    return Meta(
        labels=labels,
        label_index=label_index,
        qid_allowed_mc=qid_allowed_mc,
        qid_log_prior=qid_label_counts,
        qid_true_answer=qid_true_answer,
        exact_match_dict=exact_match_dict,
    )


# ------------------------ 後処理本体 ------------------------

def apply_postprocess(
    logits: np.ndarray,
    test_df: pd.DataFrame,
    meta: Meta,
    alpha: float = 1.5,
    beta: float = 1.0,
    gamma: float = 3.0,
    exact_hard_weight: float = 50.0,
) -> np.ndarray:
    """
    引数
    ----
    logits : (N, C) の logit（または確率→内部で log に変換）
    test_df : 必須列 ['row_id','QuestionId','MC_Answer','StudentExplanation']
    meta : build_meta_from_train の結果
    alpha : 事前分布 log P(label|Qid) の係数
    beta : キーワード加点（Misconception 単位）
    gamma : True/False 押し分け（MC_Answer 一致に応じて）
    exact_hard_weight : 完全一致辞書ヒット時に上位候補へ与える強補正（logit 加算）

    戻り値
    ------
    re_scored : (N, C) の後処理後 logits
    """
    L = logits.copy()
    L = to_logit_space(L)  # 必要に応じて log(prob) 化
    C = L.shape[1]

    # 便利なベクトル（カテゴリ True/False、Misconception 名など）
    label_cats = []
    label_mcs = []
    is_mis_label = []
    for lbl in meta.labels:
        cat, mc = split_label(lbl)
        label_cats.append(cat)
        label_mcs.append(mc)
        is_mis_label.append(cat.endswith("_Misconception"))
    label_cats = np.array(label_cats, dtype=object)
    label_mcs = np.array(label_mcs, dtype=object)
    is_mis_label = np.array(is_mis_label, dtype=bool)

    true_mask = np.array([c.startswith("True_") for c in label_cats], dtype=bool)
    false_mask = np.array([c.startswith("False_") for c in label_cats], dtype=bool)

    # 事前分布（QIDごと）とマスク適用
    re_scored = L.copy()

    row_ids = test_df["row_id"].tolist()
    qids = test_df["QuestionId"].astype(int).tolist()
    answers = test_df["MC_Answer"].astype(str).tolist()
    texts = test_df["StudentExplanation"].astype(str).map(normalize_text).tolist()

    # キーワード正規表現を Misconception -> compiled regex で用意済み（KEYWORD_HINTS）

    for i in range(L.shape[0]):
        qid = qids[i]
        ans = answers[i]
        text = texts[i]

        # 1) マスク: *_Misconception ラベルで、許容されていない Misconception を -inf に
        if qid in meta.qid_allowed_mc:
            allowed = meta.qid_allowed_mc[qid]
            bad_mask = is_mis_label & (~np.isin(label_mcs, list(allowed)))
            re_scored[i, bad_mask] = -1e30  # 実質 -inf
        else:
            # 許容未知なら何もしない（マスクなし）
            pass

        # 2) 事前分布: log P(label|Qid) を加算
        if qid in meta.qid_log_prior:
            re_scored[i, :] += alpha * meta.qid_log_prior[qid]

        # 3) True/False バイアス: 正答一致なら True_* を、非一致なら False_* を押し上げ
        if qid in meta.qid_true_answer:
            is_correct_ans = (ans == meta.qid_true_answer[qid])
            if is_correct_ans:
                re_scored[i, true_mask] += gamma
            else:
                re_scored[i, false_mask] += gamma

        # 4) キーワード加点: 文にヒットした Misconception の *_Misconception ラベルへ加点
        for mc, rx in KEYWORD_HINTS.items():
            if rx.search(text):
                hit_mask = is_mis_label & (label_mcs == mc)
                re_scored[i, hit_mask] += beta

        # 5) 完全一致辞書: 学習側で合意率の高い同一説明文があれば強補正
        if text in meta.exact_match_dict:
            lbl, agree = meta.exact_match_dict[text]
            j = meta.label_index.get(lbl, None)
            if j is not None:
                re_scored[i, j] += exact_hard_weight * max(0.0, min(1.0, agree))

    return re_scored


# ------------------------ 読み込みと提出生成 ------------------------

def read_predictions(path: str, labels_path: Optional[str] = None) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    予測ファイルを読み込み、(scores, labels, row_ids) を返す。
    - CSV: 1列目が row_id、2列目以降はラベル名（列名）＆数値
    - NPY/NPZ: (N,C) 配列。labels_path に JSON (list[str]) が必要。row_id は 0..N-1 で作成。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        # row_id 列を推定
        cand_row = [c for c in df.columns if c.lower() in ("row_id", "id")]
        if not cand_row:
            raise ValueError("predictions.csv に row_id 列が見つかりません（row_id または id を用意してください）。")
        row_col = cand_row[0]
        row_ids = df[row_col].astype(str).tolist()
        label_cols = [c for c in df.columns if c != row_col]
        if not label_cols:
            raise ValueError("predictions.csv にラベル列がありません。")
        labels = label_cols
        scores = df[label_cols].to_numpy(dtype=np.float64)
        return scores, labels, row_ids

    elif p.suffix.lower() in (".npy", ".npz"):
        arr = np.load(p, allow_pickle=False)
        if isinstance(arr, np.lib.npyio.NpzFile):
            # assume 'scores'
            if "scores" in arr:
                scores = arr["scores"]
            else:
                # 最初の配列を使う
                first_key = list(arr.files)[0]
                scores = arr[first_key]
        else:
            scores = arr
        if labels_path is None:
            raise ValueError("NPY/NPZ の場合は --labels で列順に対応する JSON (list[str]) を指定してください。")
        labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        if scores.shape[1] != len(labels):
            raise ValueError(f"スコア列数({scores.shape[1]})とラベル数({len(labels)})が一致しません。")
        row_ids = [str(i) for i in range(scores.shape[0])]
        return scores.astype(np.float64), labels, row_ids

    else:
        raise ValueError(f"未知のファイル拡張子: {p.suffix}")


def make_submission(
    rescored_logits: np.ndarray,
    labels: List[str],
    row_ids: List[str],
    topk: int = 3,
) -> pd.DataFrame:
    probs = softmax(rescored_logits, axis=1)
    topk_idx = np.argsort(-probs, axis=1)[:, :topk]
    topk_labels = [[labels[j] for j in idxs] for idxs in topk_idx]
    pred_strs = [" ".join(lbls) for lbls in topk_labels]
    sub = pd.DataFrame({"row_id": row_ids, "prediction": pred_strs})
    return sub


# ------------------------ メイン ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train.csv のパス（学習データ）")
    ap.add_argument("--test", required=True, help="test.csv のパス（row_id, QuestionId, MC_Answer, StudentExplanation を含む）")
    ap.add_argument("--pred", required=True, help="モデル予測（logits or 確率）の CSV / NPY / NPZ")
    ap.add_argument("--labels", default=None, help="NPY/NPZ の場合に必要な、ラベル名リスト JSON (list[str])")
    ap.add_argument("--out", required=True, help="出力 submission.csv のパス")

    ap.add_argument("--alpha", type=float, default=1.5, help="事前分布の強さ（log prior の係数）")
    ap.add_argument("--beta", type=float, default=1.0, help="キーワード加点の強さ")
    ap.add_argument("--gamma", type=float, default=3.0, help="True/False 押し分けの強さ")
    ap.add_argument("--exact-threshold", type=float, default=0.9, help="完全一致辞書の学習側合意率しきい値（これ以上で採用）")
    ap.add_argument("--exact-hard-weight", type=float, default=50.0, help="完全一致ヒット時の強補正（logit 加算値）")
    ap.add_argument("--topk", type=int, default=3, help="提出で返す上位件数（MAP@3なら3のまま）")

    args = ap.parse_args()

    # 予測読み込み
    scores, pred_labels, row_ids = read_predictions(args.pred, args.labels)

    # test 読み込み
    test_df = pd.read_csv(args.test)
    # 必須列チェック
    for col in ["row_id", "QuestionId", "MC_Answer", "StudentExplanation"]:
        if col not in test_df.columns:
            raise ValueError(f"test.csv に {col} 列が必要です。")
    # row_id の順序を pred と一致させる（pred の row_ids に従って並び替え）
    test_df = test_df.set_index(test_df["row_id"].astype(str))
    try:
        test_df = test_df.loc[row_ids]
    except KeyError:
        # row_id が連番などで合わない場合は警告して共通部分で進める
        common = [rid for rid in row_ids if rid in test_df.index]
        if not common:
            raise RuntimeError("predictions と test の row_id が一致しません。整合するファイルを指定してください。")
        # 対応部分のみ処理
        sel_idx = [i for i, rid in enumerate(row_ids) if rid in test_df.index]
        scores = scores[sel_idx]
        row_ids = [row_ids[i] for i in sel_idx]
        test_df = test_df.loc[row_ids]

    # メタ構築（train に基づく）
    meta = build_meta_from_train(args.train, pred_labels, exact_threshold=args.exact_threshold)

    # 後処理の適用
    rescored = apply_postprocess(
        logits=scores,
        test_df=test_df.reset_index(drop=True),
        meta=meta,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        exact_hard_weight=args.exact_hard_weight,
    )

    # 提出生成
    sub = make_submission(rescored, pred_labels, row_ids, topk=args.topk)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote: {out_path}  (rows={len(sub)})")


if __name__ == "__main__":
    main()
