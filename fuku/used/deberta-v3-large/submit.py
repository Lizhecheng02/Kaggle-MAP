#!/usr/bin/env python3
"""
submit.py: テストデータに対してモデル推論を行い、submission.csvを生成するスクリプト
"""
import os
import joblib
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, DebertaV2ForSequenceClassification
from datasets import Dataset

import config
from utils import (
    load_data,
    feature_engineer,
    prepare_correct_df,
    format_input,
    tokenize,
)


def main():
    # 環境変数設定
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
    # 設定値参照
    model_dir = config.MODEL_DIR

    # テストデータ読み込み
    test = load_data(config.TEST_CSV_PATH)
    test = feature_engineer(test)

    # 正解選択肢のマッピング
    train_for_correct = load_data(config.TRAIN_CSV_PATH)
    correct = prepare_correct_df(train_for_correct)
    test = test.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
    test['is_correct'] = test['is_correct'].fillna(0)

    # テキスト整形
    test['text'] = test.apply(format_input, axis=1)

    # データセット作成
    ds_test = Dataset.from_pandas(test[['text']])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ds_test = ds_test.map(lambda batch: tokenize(batch, tokenizer), batched=True)
    ds_test.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # モデルとラベルエンコーダ読み込み
    model = DebertaV2ForSequenceClassification.from_pretrained(model_dir)
    # ラベルエンコーダ読み込み
    le = joblib.load(f"{config.DIR}/label_encoder.joblib")

    # 推論設定
    training_args = TrainingArguments(report_to="none")
    trainer = Trainer(model=model, args=training_args)
    predictions = trainer.predict(ds_test)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()

    # Top3予測
    top3 = np.argsort(-probs, axis=1)[:, :3]
    flat_top3 = top3.flatten()
    decoded_labels = le.inverse_transform(flat_top3)
    top3_labels = decoded_labels.reshape(top3.shape)
    joined_preds = [" ".join(row) for row in top3_labels]

    # 提出ファイル作成
    sub = pd.DataFrame({
        "row_id": test.row_id.values,
        "Category:Misconception": joined_preds
    })
    sub.to_csv("submission.csv", index=False)
    print("submission.csv を保存しました")


if __name__ == '__main__':
    main()
