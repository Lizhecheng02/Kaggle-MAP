import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import create_submission


class DummyPred:
    def __init__(self, arr):
        self.predictions = np.asarray(arr)


def test_create_submission_with_question_choices():
    # Label space: include three allowed for 31772 and one disallowed with big logit
    classes = [
        '31772:Correct:NA',          # idx 0 -> 0.1
        'Q:Neither:NA',              # idx 1 -> 0.2
        '31772:Misconception:Incomplete',  # idx 2 -> 0.3 (top among allowed)
        '32829:Misconception:Adding_terms' # idx 3 -> 10.0 (NOT allowed for 31772)
    ]
    le = LabelEncoder()
    le.fit(classes)

    # one sample logits
    logits = np.array([[0.1, 0.2, 0.3, 10.0]])
    pred = DummyPred(logits)

    test_df = pd.DataFrame({
        'row_id': [0],
        'QuestionId': ['31772'],
        'is_correct': [True],
    })

    sub = create_submission(pred, test_df, le)

    assert 'Category:Misconception' in sub.columns
    # top1 must be True_Misconception:Incomplete (the disallowed highest logit must be excluded)
    assert sub['Category:Misconception'].iloc[0].split()[0] == 'True_Misconception:Incomplete'


def test_create_submission_fallback_when_no_mapping():
    # No mapping for qid 99999 -> fallback to global top-3
    classes = ['A:Correct:NA', 'Q:Neither:NA', 'A:Misconception:X']
    le = LabelEncoder().fit(classes)
    logits = np.array([[0.9, 0.05, 0.04]])
    pred = DummyPred(logits)
    test_df = pd.DataFrame({
        'row_id': [0],
        'QuestionId': ['99999'],
        'is_correct': [False],
    })
    sub = create_submission(pred, test_df, le)
    # top1 must be False_Correct:NA (since global top is class 0)
    assert sub['Category:Misconception'].iloc[0].split()[0] == 'False_Correct:NA'

