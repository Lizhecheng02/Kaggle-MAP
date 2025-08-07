Overview
In this competition, you’ll develop an NLP model driven by ML to accurately predict students’ potential math misconceptions based on student explanations in open-ended responses. This solution will suggest candidate misconceptions for these explanations, making it easier for teachers to identify and address students’ incorrect thinking, which is critical to improving student math learning.

Start

18 hours ago
Close
3 months to go
Merger & Entry
Description
Students are often asked to explain their mathematical reasoning. These explanations provide rich insight into student thinking and often reveal underlying misconceptions (systematic incorrect ways of thinking).

For example, students often think 0.355 is larger than 0.8 because they incorrectly apply their knowledge of whole numbers to decimals, reasoning that 355 is greater than 8. Students develop a range of misconceptions in math, sometimes because they incorrectly apply prior knowledge to new content and sometimes because they are trying to make sense of new information but misunderstand it. To read more about these definitions and framework, please see the linked report here.This competition aims to explore such possibilities by encouraging participants to develop models that can distinguish between these types of conceptual errors in students’ responses, paving the way for improved feedback and better learning outcomes.

Tagging students’ explanations as containing potential misconceptions is valuable for diagnostic feedback but is time-consuming and challenging to scale. Misconceptions can be subtle, vary in specificity, and evolve as new patterns in student reasoning emerge.
Initial efforts to use pre-trained language models have not been successful, likely due to the complexity of the mathematical content in the questions. Therefore, a more efficient and consistent approach is needed to streamline the tagging process and enhance the overall quality.

The MAP (Misconception Annotation Project) competition challenges you to develop a Natural Language Processing (NLP) model driven by Machine Learning (ML) that predicts students’ potential math misconceptions based on student explanations. The goal is to create a model that identifies potential math misconceptions that generalize across different problems.
Your work could help improve the understanding and management of misconceptions, enhancing the educational experience for both students and teachers.

Vanderbilt University and The Learning Agency have partnered with Kaggle to host this competition.

Acknowledgments
Vanderbilt University and The Learning Agency would like to thank the Gates Foundation and the Walton Family Foundation for their support in making this work possible, as well as Eedi for providing the data. Eedi is an edtech platform that helps students ages 9 to 16 master math by identifying and resolving misconceptions using diagnostic questions, AI-powered insights, and access to live one-on-one tutoring to boost understanding and confidence.

Walton Logo

Evaluation
Submissions are evaluated according to the Mean Average Precision @ 3 (MAP@3):


where
 is the number of observations,
 is the precision at cutoff
,
 is the number predictions submitted per observation, and
 is an indicator function equaling 1 if the item at rank
 is a relevant (correct) label, zero otherwise.

Once a correct label has been scored for an observation, that label is no longer considered relevant for that observation, and additional predictions of that label are skipped in the calculation. For example, if the correct label is A for an observation, the following predictions all score an average precision of 1.0.

Dataset Description
On Eedi, students answer Diagnostic Questions (DQs), which are multiple-choice questions featuring one correct answer and three incorrect answers, known as distractors. After responding with a multiple-choice selection, students were sometimes asked to provide a written explanation justifying their selected answer. These explanations are the primary focus of the MAP dataset and are to be used to identify and address potential misconceptions in students’ reasoning.

The goal of the competition is to develop a model that performs 3 steps:

Determines whether the selected answer is correct. (True or False in Category; e.g., True_Correct)
Assesses whether the explanation contains a misconception. (Correct, Misconception, or Neither in Category; e.g., True_Correct)
Identifies the specific misconception present, if any.
The Diagnostic Questions were presented in image format on the Eedi platform. All question content, including mathematical expressions, has been extracted via a human-in-the-loop OCR process to ensure accuracy.

Files
[train/test].csv

QuestionId - Unique question identifier.
QuestionText - The text of the question.
MC_Answer - The multiple-choice answer the student selected.
StudentExplanation - A student's explanation for choosing a specific multiple-choice answer.
Category - [train only] A classification of the relationship between a student's multiple-choice answer and their explanation (e.g., True_Misconception, which indicates a correct multiple-choice answer selection accompanied by an explanation that reveals a misconception).
Misconception - [train only] The math misconception identified in the student's explanation for answers. Only applicable when Category contains a misconception, otherwise is 'NA'.
sample_submission.csv - A submission file in the correct format.

Category:Misconception - The predicted classification Category concatenated with the Misconception by a colon (:). Up to three predictions can be made, separated by a space.
The re-run test data contains approximately 40,000 rows.

データセットサイズ: (36696, 7)

=== カラム情報 ===
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 36696 entries, 0 to 36695
Data columns (total 7 columns):
 #   Column              Non-Null Count  Dtype
---  ------              --------------  -----
 0   row_id              36696 non-null  int64
 1   QuestionId          36696 non-null  int64
 2   QuestionText        36696 non-null  object
 3   MC_Answer           36696 non-null  object
 4   StudentExplanation  36696 non-null  object
 5   Category            36696 non-null  object
 6   Misconception       9860 non-null   object
dtypes: int64(2), object(5)
memory usage: 2.0+ MB
None

=== 基本統計 ===
             row_id     QuestionId
count  36696.000000   36696.000000
mean   18347.500000   46356.825104
std    10593.367076   25617.723392
min        0.000000   31772.000000
25%     9173.750000   31777.000000
50%    18347.500000   32833.000000
75%    27521.250000   33474.000000
max    36695.000000  109465.000000

=== Category分布 ===
Category
True_Correct           14802
False_Misconception     9457
False_Neither           6542
True_Neither            5265
True_Misconception       403
False_Correct            227
Name: count, dtype: int64

Categoryユニーク数: 6

=== Misconception分布（上位20）===
Misconception
NA                         26836
Incomplete                  1454
Additive                     929
Duplication                  704
Subtraction                  620
Positive                     566
Wrong_term                   558
Irrelevant                   497
Wrong_fraction               418
Inversion                    414
Mult                         353
Denominator-only_change      336
Whole_numbers_larger         329
Adding_across                307
WNB                          299
Tacking                      290
Unknowable                   282
Wrong_Fraction               273
SwapDividend                 206
Scale                        179
Name: count, dtype: int64

Misconceptionユニーク数: 36

=== CategoryごとのMisconception有無 ===
Category
False_Correct            0.0
False_Misconception    100.0
False_Neither            0.0
True_Correct             0.0
True_Misconception     100.0
True_Neither             0.0
Name: Misconception, dtype: float64

=== QuestionIdごとの統計 ===
       answer_count  unique_categories  unique_misconceptions
count     15.000000          15.000000              15.000000
mean    2446.400000           5.866667               3.600000
std     1096.354596           0.351866               0.632456
min      673.000000           5.000000               3.000000
25%     1654.000000           6.000000               3.000000
50%     2610.000000           6.000000               4.000000
75%     3079.500000           6.000000               4.000000
max     4857.000000           6.000000               5.000000

=== テキスト長さ統計 ===
MC_Answer長さ: 平均12.9, 中央値9.0
StudentExplanation長さ: 平均70.0, 中央値60.0

=== よくあるミスコンセプション Top10 ===
1. Incomplete: 1454件
2. Additive: 929件
3. Duplication: 704件
4. Subtraction: 620件
5. Positive: 566件
6. Wrong_term: 558件
7. Irrelevant: 497件
8. Wrong_fraction: 418件
9. Inversion: 414件
10. Mult: 353件

=== ターゲットのユニーク数: 65 ===
ターゲット分布（上位20）:
target
True_Correct:NA                                14802
False_Neither:NA                                6542
True_Neither:NA                                 5265
False_Misconception:Incomplete                  1446
False_Misconception:Additive                     891
False_Misconception:Duplication                  698
False_Misconception:Subtraction                  618
False_Misconception:Positive                     564
False_Misconception:Wrong_term                   550
False_Misconception:Wrong_fraction               412
False_Misconception:Irrelevant                   409
False_Misconception:Inversion                    409
False_Misconception:Mult                         345
False_Misconception:Denominator-only_change      332
False_Misconception:Whole_numbers_larger         328
False_Misconception:Adding_across                306
False_Misconception:WNB                          291
False_Misconception:Unknowable                   282
False_Misconception:Wrong_Fraction               273
False_Correct:NA                                 227
Name: count, dtype: int64


● LoRAの各パラメータをチューニングすると、以下のような影響があります：

  1. r (ランク) = 16

  - 低い値 (4-8): パラメータ数が少なく、学習が高速でメモリ効率的だが、表現力が制限される
  - 高い値 (32-64): より複雑なパターンを学習できるが、過学習のリスクとメモリ使用量が増加
  - 影響: モデルの表現力と計算コスト・メモリ使用量のトレードオフ

  2. lora_alpha (スケーリング) = 32

  - 計算式: 実際のスケーリング = lora_alpha / r (現在は32/16=2)
  - 低い値: 更新が控えめになり、元のモデルの知識をより保持
  - 高い値: より大きな更新が可能で、タスク固有の学習が強化される
  - 影響: 学習率のような効果を持ち、更新の強度を制御

  3. target_modules

  - 現在の設定: Attention層とMLP層の主要な投影行列をすべてカバー
  - 少ないモジュール: 学習パラメータが減り、高速だが適応能力が限定的
  - 影響: どの層を微調整するかを決定し、モデルの適応範囲に直接影響

  4. lora_dropout = 0.1

  - 0に近い: 正則化が弱く、過学習しやすい
  - 高い値 (0.2-0.3): より強い正則化で汎化性能が向上する可能性
  - 影響: 過学習の防止と汎化性能のバランス

  5. bias = "none"

  - "none": バイアス項は更新されない（一般的な設定）
  - "all": すべてのバイアスを更新（パラメータ数が増加）
  - 影響: 通常は"none"で十分、特定のタスクでは"all"が有効な場合も

  6. task_type = TaskType.SEQ_CLS

  - 分類タスク用の設定（現在の設定は適切）
  - 他のタスク（因果言語モデリング等）では異なる設定が必要

  推奨調整方針:
  - まずrを8→16→32と変えて最適値を探す
  - lora_alphaはrの1-2倍程度が一般的
  - 過学習が見られたらlora_dropoutを0.1→0.2に増やす
