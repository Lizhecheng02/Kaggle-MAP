"""
utils用の設定値を管理するモジュール。

モデル予測の表記ゆれ修正（Wrong_Fraction → Wrong_fraction）に関する既知のパッチ設定を保持する。
このファイルで管理することで、コードに値をハードコードしない方針を担保する。
"""

# 置換元と置換先のラベル
WRONG_FRACTION_FROM_LABEL = "Wrong_Fraction"
WRONG_FRACTION_TO_LABEL = "Wrong_fraction"

# 既知の対象QuestionId（必要に応じて拡張可能）
WRONG_FRACTION_FIX_QIDS = [33471]

# 全行に対して置換を適用するか（デフォルトはFalseで安全側）
WRONG_FRACTION_APPLY_GLOBALLY = False

