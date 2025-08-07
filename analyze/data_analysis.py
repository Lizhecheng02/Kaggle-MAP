# -*- coding: utf-8 -*-
"""
データセット詳細分析スクリプト
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# データ読み込み
train = pd.read_csv(
    "/kaggle/input/map-charting-student-math-misunderstandings/train.csv"
)

print("データセットサイズ:", train.shape)
print("\n=== カラム情報 ===")
print(train.info())
print("\n=== 基本統計 ===")
print(train.describe())

# Categoryの分布
print("\n=== Category分布 ===")
print(train['Category'].value_counts())
print(f"\nCategoryユニーク数: {train['Category'].nunique()}")

# Misconceptionの分布
train['Misconception'] = train['Misconception'].fillna('NA')
print("\n=== Misconception分布（上位20）===")
print(train['Misconception'].value_counts().head(20))
print(f"\nMisconceptionユニーク数: {train['Misconception'].nunique()}")

# Categoryごとのデータ数
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Category分布
category_counts = train['Category'].value_counts()
ax1.bar(range(len(category_counts)), category_counts.values)
ax1.set_xticks(range(len(category_counts)))
ax1.set_xticklabels(category_counts.index, rotation=45, ha='right')
ax1.set_title('Category Distribution')
ax1.set_ylabel('Count')

# Misconceptionの有無
misconception_present = train['Misconception'].apply(lambda x: 'Present' if x != 'NA' else 'Absent').value_counts()
ax2.pie(misconception_present.values, labels=misconception_present.index, autopct='%1.1f%%')
ax2.set_title('Misconception Presence')

plt.tight_layout()
plt.savefig('category_distribution.png')

# CategoryとMisconceptionの関係分析
print("\n=== CategoryごとのMisconception有無 ===")
category_misconception = train.groupby('Category')['Misconception'].apply(lambda x: (x != 'NA').sum() / len(x) * 100)
print(category_misconception.round(2))

# QuestionIdごとの統計
print("\n=== QuestionIdごとの統計 ===")
question_stats = train.groupby('QuestionId').agg({
    'row_id': 'count',
    'Category': lambda x: x.nunique(),
    'Misconception': lambda x: x.nunique()
}).rename(columns={'row_id': 'answer_count', 'Category': 'unique_categories', 'Misconception': 'unique_misconceptions'})
print(question_stats.describe())

# MC_Answerの長さ分析
train['mc_answer_len'] = train['MC_Answer'].astype(str).str.len()
train['explanation_len'] = train['StudentExplanation'].astype(str).str.len()

print("\n=== テキスト長さ統計 ===")
print(f"MC_Answer長さ: 平均{train['mc_answer_len'].mean():.1f}, 中央値{train['mc_answer_len'].median():.1f}")
print(f"StudentExplanation長さ: 平均{train['explanation_len'].mean():.1f}, 中央値{train['explanation_len'].median():.1f}")

# カテゴリー別のexplanation長さ
fig, ax = plt.subplots(figsize=(10, 6))
train.boxplot(column='explanation_len', by='Category', ax=ax, rot=45)
ax.set_ylabel('Explanation Length')
ax.set_title('Explanation Length by Category')
plt.suptitle('')
plt.tight_layout()
plt.savefig('explanation_length_by_category.png')

# よくあるミスコンセプションのトップ10
top_misconceptions = train[train['Misconception'] != 'NA']['Misconception'].value_counts().head(10)
print("\n=== よくあるミスコンセプション Top10 ===")
for i, (misconception, count) in enumerate(top_misconceptions.items(), 1):
    print(f"{i}. {misconception}: {count}件")

# ターゲット（Category:Misconception）の分布
train['target'] = train['Category'] + ':' + train['Misconception']
print(f"\n=== ターゲットのユニーク数: {train['target'].nunique()} ===")
print("ターゲット分布（上位20）:")
print(train['target'].value_counts().head(20))