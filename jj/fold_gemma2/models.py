import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self, gamma: float = 2.0, alpha=None, reduction: str = "mean", eps: float = 1e-8
    ):
        """
        gamma: focusing parameter (>=0). 2.0 is common.
        alpha: None, scalar float, or 1D tensor of shape [num_classes] for per-class weights.
        reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.gamma = gamma
        if isinstance(alpha, (list, tuple)):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer(
            "alpha", alpha if isinstance(alpha, torch.Tensor) else None
        )
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: [B, C]
        targets: [B] (class indices 0..C-1)
        """
        # Compute log-prob and prob
        log_probs = F.log_softmax(logits, dim=-1)  # [B, C]
        probs = log_probs.exp()  # [B, C]

        # Gather the true-class probabilities and log-probs
        targets = targets.long()
        pt = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [B]
        log_pt = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(
            -1
        )  # [B]

        # Alpha weighting
        if self.alpha is None:
            at = 1.0
        elif self.alpha.numel() == 1:
            at = self.alpha.item()
        else:
            # Per-class alpha vector
            at = self.alpha.to(logits.device).gather(0, targets)  # [B]

        # Focal modulation
        focal_weight = (1 - pt).clamp(min=0, max=1) ** self.gamma  # [B]

        loss = -at * focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LLMForSequenceClassification(nn.Module):
    def __init__(self, base_model, num_labels, alpha_vec_or_none=None):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.num_labels = num_labels
        
        # Handle multimodal Mistral3 config structure
        if hasattr(self.config, 'text_config'):
            hidden_size = self.config.text_config.hidden_size
        else:
            hidden_size = self.config.hidden_size
            
        self.score = nn.Linear(hidden_size, num_labels)
        
        if alpha_vec_or_none is not None:
            self.alpha_vec = nn.Parameter(torch.tensor(alpha_vec_or_none, dtype=torch.float32))
        else:
            self.alpha_vec = None
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Remove conflicting arguments from kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['output_hidden_states', 'return_dict']}
        
        # Access the language model correctly
        if hasattr(self.base_model, 'language_model'):
            # For multimodal models, use the language model component
            language_model = self.base_model.language_model
        else:
            # For regular models
            language_model = self.base_model
            
        # Call the model correctly
        outputs = language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **filtered_kwargs
        )
        
        # Get the last hidden state
        hidden_states = outputs.last_hidden_state
        
        # Pool the hidden states
        if attention_mask is not None:
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            # Simple mean pooling
            pooled_output = hidden_states.mean(dim=1)
        
        # Get logits
        logits = self.score(pooled_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            if self.alpha_vec is not None:
                # Focal loss or weighted loss
                loss_fct = nn.CrossEntropyLoss(weight=self.alpha_vec)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {"loss": loss, "logits": logits}

    def _get_hidden_size(self):
        """Get hidden size, handling nested configs for multimodal models"""

        # For Mistral3 multimodal models, check text_config first
        if hasattr(self.config, "text_config") and hasattr(
            self.config.text_config, "hidden_size"
        ):
            return self.config.text_config.hidden_size

        # Standard config locations
        for attr_name in ["hidden_size", "dim", "model_dim", "d_model", "hidden_dim"]:
            if hasattr(self.config, attr_name):
                size = getattr(self.config, attr_name)
                if isinstance(size, int) and size > 0:
                    return size

        # Vision config as fallback (though probably not what you want for text classification)
        if hasattr(self.config, "vision_config") and hasattr(
            self.config.vision_config, "hidden_size"
        ):
            print("Warning: Using vision_config.hidden_size as fallback")
            return self.config.vision_config.hidden_size

        # Final fallback: try to get from the actual model structure
        try:
            if hasattr(self.base_model, "language_model"):
                return self.base_model.language_model.config.hidden_size
            elif hasattr(self.base_model, "text_model"):
                return self.base_model.text_model.config.hidden_size
        except:
            pass

        raise AttributeError("Could not determine hidden size from config")
