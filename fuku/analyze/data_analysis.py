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

# 問題ごとの誤解数詳細分析
print("\n=== 問題ごとの誤解数詳細分析 ===")
question_misconception_stats = train.groupby('QuestionId').agg({
    'row_id': 'count',
    'Misconception': [
        lambda x: (x != 'NA').sum(),  # 誤解がある回答数
        lambda x: x.nunique() - (1 if 'NA' in x.values else 0),  # ユニークな誤解数（NAを除く）
        lambda x: (x != 'NA').sum() / len(x) * 100  # 誤解率（%）
    ]
}).round(2)

question_misconception_stats.columns = ['total_answers', 'misconception_count', 'unique_misconceptions', 'misconception_rate_pct']
question_misconception_stats = question_misconception_stats.reset_index()

print(f"問題数: {len(question_misconception_stats)}")
print(f"誤解がある回答の平均数: {question_misconception_stats['misconception_count'].mean():.1f}")
print(f"問題あたりの平均誤解率: {question_misconception_stats['misconception_rate_pct'].mean():.1f}%")

# 誤解が最も多い問題Top10
print("\n=== 誤解が最も多い問題 Top10 ===")
top_misconception_questions = question_misconception_stats.nlargest(10, 'misconception_count')
for i, row in top_misconception_questions.iterrows():
    print(f"{row['QuestionId']}: 誤解数{int(row['misconception_count'])}, 誤解率{row['misconception_rate_pct']:.1f}%, ユニーク誤解{int(row['unique_misconceptions'])}")

# 誤解率が最も高い問題Top10
print("\n=== 誤解率が最も高い問題 Top10 ===")
top_rate_questions = question_misconception_stats.nlargest(10, 'misconception_rate_pct')
for i, row in top_rate_questions.iterrows():
    print(f"{row['QuestionId']}: 誤解率{row['misconception_rate_pct']:.1f}%, 誤解数{int(row['misconception_count'])}, 総回答数{int(row['total_answers'])}")

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

# 問題ごとの誤解数の視覚化
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 1. 誤解数のヒストグラム
ax1.hist(question_misconception_stats['misconception_count'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
ax1.set_xlabel('誤解数')
ax1.set_ylabel('問題数')
ax1.set_title('問題ごとの誤解数分布')
ax1.grid(True, alpha=0.3)

# 2. 誤解率のヒストグラム
ax2.hist(question_misconception_stats['misconception_rate_pct'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
ax2.set_xlabel('誤解率 (%)')
ax2.set_ylabel('問題数')
ax2.set_title('問題ごとの誤解率分布')
ax2.grid(True, alpha=0.3)

# 3. 誤解数と総回答数の散布図
ax3.scatter(question_misconception_stats['total_answers'], 
           question_misconception_stats['misconception_count'], 
           alpha=0.6, color='green')
ax3.set_xlabel('総回答数')
ax3.set_ylabel('誤解数')
ax3.set_title('総回答数 vs 誤解数')
ax3.grid(True, alpha=0.3)

# 4. 誤解が多い上位10問題の棒グラフ
top_10 = question_misconception_stats.nlargest(10, 'misconception_count')
ax4.bar(range(len(top_10)), top_10['misconception_count'], color='orange', alpha=0.7)
ax4.set_xlabel('問題 (ランキング)')
ax4.set_ylabel('誤解数')
ax4.set_title('誤解数上位10問題')
ax4.set_xticks(range(len(top_10)))
ax4.set_xticklabels([f'Q{i+1}' for i in range(len(top_10))])
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('question_misconception_analysis.png', dpi=300, bbox_inches='tight')

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

# 分析結果をCSVファイルに出力
print("\n=== 分析結果のCSV出力 ===")

# 1. 問題ごとの誤解統計をCSVに出力
question_misconception_stats.to_csv('question_misconception_stats.csv', index=False, encoding='utf-8')
print("問題ごとの誤解統計: question_misconception_stats.csv")

# 2. 詳細な問題別誤解情報を作成
detailed_stats = []
for question_id in train['QuestionId'].unique():
    question_data = train[train['QuestionId'] == question_id]
    misconceptions = question_data[question_data['Misconception'] != 'NA']['Misconception'].value_counts()
    
    detailed_stats.append({
        'QuestionId': question_id,
        'total_answers': len(question_data),
        'misconception_answers': (question_data['Misconception'] != 'NA').sum(),
        'unique_misconceptions': len(misconceptions),
        'most_common_misconception': misconceptions.index[0] if len(misconceptions) > 0 else 'なし',
        'most_common_count': misconceptions.iloc[0] if len(misconceptions) > 0 else 0,
        'categories': ', '.join(question_data['Category'].unique())
    })

detailed_df = pd.DataFrame(detailed_stats)
detailed_df.to_csv('question_detailed_analysis.csv', index=False, encoding='utf-8')
print("詳細な問題別分析: question_detailed_analysis.csv")

# 3. カテゴリ別誤解統計
category_stats = train.groupby('Category').agg({
    'row_id': 'count',
    'Misconception': lambda x: (x != 'NA').sum(),
    'QuestionId': 'nunique'
}).rename(columns={'row_id': 'total_answers', 'Misconception': 'misconception_count', 'QuestionId': 'unique_questions'})
category_stats['misconception_rate'] = (category_stats['misconception_count'] / category_stats['total_answers'] * 100).round(2)
category_stats.to_csv('category_misconception_stats.csv', encoding='utf-8')
print("カテゴリ別誤解統計: category_misconception_stats.csv")

# QuestionIdごとのMisconception出現回数分析
print("\n=== QuestionIdごとのMisconception出現回数 ===")
question_misconception_counts = []

for question_id in train['QuestionId'].unique():
    question_data = train[train['QuestionId'] == question_id]
    
    # NAでないMisconceptionのみをカウント
    misconceptions = question_data[question_data['Misconception'] != 'NA']['Misconception'].value_counts()
    
    for misconception, count in misconceptions.items():
        question_misconception_counts.append({
            'QuestionId': question_id,
            'Misconception': misconception,
            'Count': count
        })

# DataFrameに変換
question_misconception_df = pd.DataFrame(question_misconception_counts)

# 結果を表示（上位20件）
print(f"QuestionId x Misconceptionの組み合わせ数: {len(question_misconception_df)}")
print("\n出現回数上位20件:")
top_combinations = question_misconception_df.nlargest(20, 'Count')
for i, row in top_combinations.iterrows():
    print(f"{row['QuestionId']} - {row['Misconception']}: {row['Count']}回")

# CSVファイルに出力
question_misconception_df = question_misconception_df.sort_values(['QuestionId', 'Count'], ascending=[True, False])
question_misconception_df.to_csv('question_misconception_counts.csv', index=False, encoding='utf-8')
print("\nQuestionIdごとのMisconception出現回数: question_misconception_counts.csv")

# Misconceptionごとの出現QuestionId数も分析
misconception_question_counts = question_misconception_df.groupby('Misconception').agg({
    'QuestionId': 'nunique',
    'Count': 'sum'
}).rename(columns={'QuestionId': 'unique_questions', 'Count': 'total_count'}).sort_values('total_count', ascending=False)

print(f"\n=== Misconceptionごとの出現統計（上位15件）===")
for i, (misconception, row) in enumerate(misconception_question_counts.head(15).iterrows(), 1):
    print(f"{i}. {misconception}")
    print(f"   - 出現問題数: {row['unique_questions']}問題")
    print(f"   - 総出現回数: {row['total_count']}回")

misconception_question_counts.to_csv('misconception_summary_stats.csv', encoding='utf-8')
print("\nMisconception統計サマリー: misconception_summary_stats.csv")

print("\n=== 分析完了 ===")
print("出力ファイル:")
print("- question_misconception_stats.csv: 問題ごとの誤解統計")
print("- question_detailed_analysis.csv: 詳細な問題別分析")
print("- category_misconception_stats.csv: カテゴリ別誤解統計")
print("- question_misconception_counts.csv: QuestionIdごとのMisconception出現回数")
print("- misconception_summary_stats.csv: Misconception統計サマリー")
print("- question_misconception_analysis.png: 誤解数分析グラフ")
print("- category_distribution.png: カテゴリ分布グラフ")
print("- explanation_length_by_category.png: カテゴリ別説明長さ")