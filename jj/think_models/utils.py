import pandas as pd


def prepare_correct_answers(df: pd.DataFrame) -> pd.DataFrame:

    idx = df.apply(lambda x: x.Category.split('_')[0]=='True', axis=1)
    correct = df.loc[idx].copy()
    correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
    correct = correct.sort_values('c', ascending=False)
    correct = correct.drop_duplicates(['QuestionId'])[['QuestionId', 'MC_Answer']]
    correct['is_correct'] = 1
    return correct