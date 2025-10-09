import torch
from data_collator import DataCollatorWithPadding


class DummyTokenizer:
    pad_token_id = 0


def test_data_collator_pads_and_maps_labels():
    collator = DataCollatorWithPadding(tokenizer=DummyTokenizer(), max_length=None)
    features = [
        {"input_ids": [10, 11, 12], "attention_mask": [1, 1, 1], "label": 2},
        {"input_ids": [20, 21, 22, 23, 24], "attention_mask": [1, 1, 1, 1, 1], "label": 1},
    ]

    batch = collator(features)

    assert "labels" in batch and "label" not in batch
    assert isinstance(batch["labels"], torch.Tensor)

    # 最大長に揃っていること
    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"].shape == (2, 5)

    # 先頭要素は後ろ2つがパディング（0）
    assert batch["input_ids"][0, -2:].tolist() == [0, 0]
    assert batch["attention_mask"][0, -2:].tolist() == [0, 0]

