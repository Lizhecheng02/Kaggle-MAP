"""
カスタムデータコレーター for GPT-OSS-20Bモデル
"""
import torch
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: int = None
    pad_to_multiple_of: int = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        labels = None
        if "label" in features[0]:
            labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
            features = [{k: v for k, v in f.items() if k != "label"} for f in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if labels is not None:
            batch["labels"] = labels
        return batch
