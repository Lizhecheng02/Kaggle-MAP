"""
マルチタスク学習モデルの実装
"""

import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class MultiTaskDebertaModel(DebertaV2PreTrainedModel):
    """
    3つのタスクを同時に学習するマルチタスクモデル:
    1. 正解/不正解の判定 (2クラス分類)
    2. カテゴリ分類 (Correct/Misconception/Neither - 3クラス分類)
    3. 具体的なMisconception分類 (36クラス分類)
    """
    
    def __init__(self, config, num_misconceptions=36):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_misconceptions = num_misconceptions
        
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # タスク別の分類ヘッド
        self.correctness_classifier = nn.Linear(config.hidden_size, 2)  # 正解/不正解
        self.category_classifier = nn.Linear(config.hidden_size, 3)     # Correct/Misconception/Neither
        self.misconception_classifier = nn.Linear(config.hidden_size, num_misconceptions)  # 具体的なMisconception
        
        # 最終的な予測のための結合層
        combined_size = 2 + 3 + num_misconceptions
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_labels)
        )
        
        # 各タスクの重み（学習可能）
        self.task_weights = nn.Parameter(torch.ones(3))
        
        # Initialize weights
        self.post_init()
    
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
        correctness_labels=None,
        category_labels=None,
        misconception_labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # DeBERTaエンコーダーの出力を取得
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # [CLS]トークンの表現を取得
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # 各タスクの予測
        correctness_logits = self.correctness_classifier(pooled_output)
        category_logits = self.category_classifier(pooled_output)
        misconception_logits = self.misconception_classifier(pooled_output)
        
        # 予測を結合
        combined_features = torch.cat([
            correctness_logits,
            category_logits,
            misconception_logits
        ], dim=-1)
        
        # 最終予測
        logits = self.fusion_layer(combined_features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            main_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            # 補助タスクの損失
            auxiliary_losses = []
            
            if correctness_labels is not None:
                correctness_loss = loss_fct(correctness_logits.view(-1, 2), correctness_labels.view(-1))
                auxiliary_losses.append(correctness_loss)
            
            if category_labels is not None:
                category_loss = loss_fct(category_logits.view(-1, 3), category_labels.view(-1))
                auxiliary_losses.append(category_loss)
            
            if misconception_labels is not None:
                # Misconceptionがない場合（NA）は損失計算から除外
                mask = misconception_labels >= 0
                if mask.any():
                    misconception_loss = loss_fct(
                        misconception_logits[mask],
                        misconception_labels[mask]
                    )
                    auxiliary_losses.append(misconception_loss)
            
            # 重み付き損失の計算
            if auxiliary_losses:
                # タスクの重みを正規化
                normalized_weights = torch.softmax(self.task_weights, dim=0)
                
                # メインタスクの重みを高く設定
                main_weight = 0.7
                aux_weight = 0.3
                
                weighted_aux_loss = sum(
                    normalized_weights[i] * loss 
                    for i, loss in enumerate(auxiliary_losses)
                )
                
                loss = main_weight * main_loss + aux_weight * weighted_aux_loss
            else:
                loss = main_loss
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FocalLoss(nn.Module):
    """クラス不均衡に対処するためのFocal Loss"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            focal_loss = self.alpha[targets] * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss