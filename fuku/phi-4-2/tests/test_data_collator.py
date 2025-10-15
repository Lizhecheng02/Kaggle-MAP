import torch
from data_collator import DataCollatorWithPadding


class DummyTokenizer:
    pad_token_id = 0


def test_data_collator_with_hier_labels():
    collator = DataCollatorWithPadding(tokenizer=DummyTokenizer())
    features = [
        {
            'input_ids': [1,2,3],
            'attention_mask': [1,1,1],
            'label': 0,
            'label_cat': 1,
            'label_mc': 0,
        },
        {
            'input_ids': [4,5],
            'attention_mask': [1,1],
            'label': 3,
            'label_cat': 2,
            'label_mc': 2,
        }
    ]

    batch = collator(features)
    assert 'input_ids' in batch and 'attention_mask' in batch
    assert 'labels' in batch and 'labels_cat' in batch and 'labels_mc' in batch
    assert batch['input_ids'].shape == (2, 3)
    assert batch['attention_mask'].shape == (2, 3)
    assert batch['labels'].dtype == torch.long
    assert batch['labels_cat'].dtype == torch.long
    assert batch['labels_mc'].dtype == torch.long

