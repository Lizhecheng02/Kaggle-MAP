import unittest
import pandas as pd

from utils import wrong_fraction_decoder


class TestWrongFractionDecoder(unittest.TestCase):
    def setUp(self):
        # submission DataFrame のモック
        self.submission = pd.DataFrame(
            {
                "row_id": [0, 1, 2],
                "Category:Misconception": [
                    "Wrong_Fraction Conceptual_Error Calculation_Error",
                    "Calculation_Error Wrong_Fraction Other_Label",
                    "Other_Label Calculation_Error Conceptual_Error",
                ],
            }
        )

        # test_data DataFrame のモック
        self.test_data = pd.DataFrame(
            {
                "row_id": [0, 1, 2],
                "QuestionId": [33471, 10000, 33471],
            }
        )

    def test_replace_only_target_qids(self):
        out = wrong_fraction_decoder(self.submission, self.test_data)
        # row_id 0 と 2 は QuestionId=33471 → 置換対象
        self.assertEqual(
            out.loc[0, "Category:Misconception"],
            "Wrong_fraction Conceptual_Error Calculation_Error",
        )
        self.assertEqual(
            out.loc[2, "Category:Misconception"],
            "Other_Label Calculation_Error Conceptual_Error",
        )
        # row_id 1 は対象外 → 変更なし
        self.assertEqual(
            out.loc[1, "Category:Misconception"],
            "Calculation_Error Wrong_Fraction Other_Label",
        )

    def test_apply_globally_true(self):
        out = wrong_fraction_decoder(self.submission, self.test_data, apply_globally=True)
        # 全行でWrong_FractionがWrong_fractionに置換される
        self.assertEqual(
            out.loc[0, "Category:Misconception"],
            "Wrong_fraction Conceptual_Error Calculation_Error",
        )
        self.assertEqual(
            out.loc[1, "Category:Misconception"],
            "Calculation_Error Wrong_fraction Other_Label",
        )

    def test_without_test_data_no_change(self):
        out = wrong_fraction_decoder(self.submission, test_data=None)
        # test_dataが無い、かつapply_globally=False(既定) → 何も変えない
        pd.testing.assert_frame_equal(out, self.submission)


if __name__ == "__main__":
    unittest.main()

