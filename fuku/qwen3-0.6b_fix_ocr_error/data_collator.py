"""
カスタムデータコレーター for Qwen3モデル
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
        # バッチの最大長を取得
        max_length = max(len(feature["input_ids"]) for feature in features)
        
        # パディング
        batch = {}
        for key in features[0].keys():
            if key == "label":
                # ラベルはパディング不要
                batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
            elif key in ["input_ids", "attention_mask"]:
                # input_idsとattention_maskをパディング
                padded = []
                for feature in features:
                    # tensorをlistに変換
                    if torch.is_tensor(feature[key]):
                        feature_list = feature[key].tolist()
                    else:
                        feature_list = feature[key]
                    
                    remainder = [self.tokenizer.pad_token_id if key == "input_ids" else 0] * (max_length - len(feature_list))
                    padded_feature = feature_list + remainder
                    padded.append(padded_feature)
                batch[key] = torch.tensor(padded, dtype=torch.long)
        
        # labelsフィールドを追加（Trainerが期待するため）
        if "label" in batch:
            batch["labels"] = batch.pop("label")  # labelを削除してlabelsに変更
            
        return batch