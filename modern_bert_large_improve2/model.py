"""
Multi-sample dropout headを持つModernBERTモデル
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class ModernBERTWithMultiDropout(nn.Module):
    """Multi-sample dropout headを持つModernBERTモデル"""
    
    def __init__(self, model_name, num_labels, num_dropout_heads=5, dropout_p=0.5):
        super(ModernBERTWithMultiDropout, self).__init__()
        
        # ベースモデルの読み込み
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config
        hidden_size = self.config.hidden_size
        
        # Multi-sample dropout heads
        self.num_dropout_heads = num_dropout_heads
        self.dropout_p = dropout_p
        
        # 複数のdropout layerを作成
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_p) for _ in range(num_dropout_heads)
        ])
        
        # 分類用のlinear layer
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # 重みの初期化
        self._init_weights()
        
    def _init_weights(self):
        """重みの初期化"""
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # BERTエンコーダーの出力を取得
        # ModernBERTはtoken_type_idsをサポートしないため、除外
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # [CLS]トークンの表現を取得
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        
        # Multi-sample dropoutの適用
        if self.training:
            # 訓練時：各dropout headからlogitsを取得して平均化
            logits_list = []
            for dropout in self.dropouts:
                dropped = dropout(cls_output)
                logits = self.classifier(dropped)
                logits_list.append(logits)
            
            # logitsの平均を計算
            logits = torch.stack(logits_list, dim=0).mean(dim=0)
        else:
            # 推論時：dropoutなしで予測
            logits = self.classifier(cls_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def save_pretrained(self, save_directory):
        """モデルの保存"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")
        self.bert.config.save_pretrained(save_directory)
        
    @classmethod
    def from_pretrained(cls, save_directory, model_name, num_labels, num_dropout_heads=5, dropout_p=0.5):
        """保存されたモデルの読み込み"""
        import os
        from safetensors.torch import load_file
        
        model = cls(model_name, num_labels, num_dropout_heads, dropout_p)
        
        # safetensorsファイルが存在する場合はそちらを優先
        safetensors_path = f"{save_directory}/model.safetensors"
        pytorch_path = f"{save_directory}/pytorch_model.bin"
        
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path)
        else:
            raise FileNotFoundError(f"No model file found in {save_directory}")
        
        model.load_state_dict(state_dict)
        return model