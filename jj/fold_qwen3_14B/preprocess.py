import re
import pandas as pd


def fix_question_text_89443(df: pd.DataFrame) -> pd.DataFrame:
    new_question_text = """
Question: What number belongs in the box?
\((-8)-(-5)=x\)
"""

    df.loc[df["QuestionId"] == 89443, "QuestionText"] = new_question_text

    df["QuestionText"] = df["QuestionText"].apply(lambda x: x.strip())

    return df


def fix_question_category_false_neither_31778(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[
        (df["QuestionId"] == 31778)
        & (df["MC_Answer"] == "\( 6 \)")
        & (df["Category"].str.contains("False")),
        "Category",
    ] = "True_Neither"

    return df


def fix_question_category_wrong_31778(df: pd.DataFrame) -> pd.DataFrame:
    # Row: 14280
    # Category: True_Neither
    # Explanation: Because 10 is 2 / 3 of 15, and 2 is 6.
    # Possibly Misconception Incomplete?
    df.loc[(df["row_id"] == 14280), "MC_Answer"] = "\( 6 \)"
    df.loc[(df["row_id"] == 14280), "Category"] = "True_Misconception"
    df.loc[(df["row_id"] == 14280), "Misconception"] = "Irrelevant"

    # Row: 14305
    # Category: True_Correct
    # Explanation: I divided 9/15 by 3, then got 3/5 and timsed it by 2 and got 6/10.
    df.loc[(df["row_id"] == 14305), "MC_Answer"] = "\( 6 \)"

    # Row: 14321
    # Category: True_Correct
    # Explanation: I think it's C because 6/10 is the same as 9/15.
    df.loc[(df["row_id"] == 14321), "MC_Answer"] = "\( 6 \)"

    # Row: 14335
    # Category: True_Correct
    # Explanation: Il believe that is the ansewer because I calculatted iti.
    df.loc[(df["row_id"] == 14335), "Category"] = "False_Neither"

    # Row: 14338
    # Category: True_Correct
    # Explanation: It is six because they are both equal to 3over5.
    df.loc[(df["row_id"] == 14338), "MC_Answer"] = "\( 6 \)"

    # Row: 14352
    # Category: True_Correct
    # Explanation: To get a denominator of 10, we need to divide by 3 and multiply by 2. Then, 9/15=3/5=6/10, so A = 6.
    df.loc[(df["row_id"] == 14352), "MC_Answer"] = "\( 6 \)"

    # Row: 14355
    # Category: True_Neither
    # Explanation: You have to change the denominator to 150 then you will get the answer.
    # Possibly Misconception Incomplete?
    df.loc[(df["row_id"] == 14355), "Category"] = "False_Neither"

    # Row: 14403
    # Category: True_Correct
    # Explanation: i think this is because 9/15=18/30 and 6/10 =18-30.
    df.loc[(df["row_id"] == 14403), "MC_Answer"] = "\( 6 \)"
    df.loc[(df["row_id"] == 14403), "Category"] = "True_Misconception"
    df.loc[(df["row_id"] == 14403), "Misconception"] = "WNB"

    # Row: 14407
    # Category: False_Neither
    # Explanation: if you simplify it to 3/5 then you get 9/15.
    df.loc[(df["row_id"] == 14407), "Category"] = "False_Neither"

    # Row: 14412
    # Category: True_Neither
    # Explanation: since 9 - 3 = 6h are so i't must be these ohne!
    df.loc[(df["row_id"] == 14412), "Category"] = "False_Misconception"
    df.loc[(df["row_id"] == 14412), "Misconception"] = "WNB"

    # Row: 14413
    # Category: True_Correct
    # Explanation: so the common denominator is 30 and the product of 15x2=30 so 9x2, which is a multiple of 9, is 18 and 10x3 =30, so ax3, which we know is 18, is 18. therefore, a=6.
    df.loc[(df["row_id"] == 14413), "MC_Answer"] = "\( 6 \)"

    # Row: 14418
    # Category: True_Misconception
    # Explanation: this is because the top numbers go up in threes, and the bottom number go down in fives.
    df.loc[(df["row_id"] == 14418), "MC_Answer"] = "\( 6 \)"

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    _df = df.copy()

    # Create 'NA' Misconception
    _df = _df.fillna("NA")

    # Clean Data
    _df["StudentExplanation"] = _df["StudentExplanation"].apply(
        lambda x: x.strip().strip(".")
    )

    # _df['FixedStudentExplanation'] = replace_misspelled_words(df['StudentExplanation'].tolist(), misspelled)

    _df = fix_question_text_89443(_df)
    _df = fix_question_category_false_neither_31778(_df)
    _df = fix_question_category_wrong_31778(_df)

    _df["Correct"] = _df.Category.apply(lambda x: True if "True" in x else False)

    # Remove unnecessary duplicates
    _df = _df.drop_duplicates(
        subset=[
            "QuestionId",
            "MC_Answer",
            "StudentExplanation",
            "Category",
            "Misconception",
            "Correct",
        ]
    ).reset_index(drop=True)

    return _df
