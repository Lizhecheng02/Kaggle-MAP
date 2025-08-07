import pandas as pd

# データ読み込み
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train['Misconception'] = train['Misconception'].fillna('NA')

# Category分布
print('=== Category分布 ===')
print(train['Category'].value_counts())
print()

# CategoryごとのMisconception数
print('=== CategoryごとのMisconception種類数 ===')
for cat in train['Category'].unique():
    misc_count = train[train['Category'] == cat]['Misconception'].nunique()
    print(f'{cat}: {misc_count}種類')
print()

# Misconceptionを持つCategoryのみ
print('=== Misconceptionが存在するCategory ===')
has_misc = train[train['Misconception'] != 'NA']['Category'].unique()
print(has_misc)