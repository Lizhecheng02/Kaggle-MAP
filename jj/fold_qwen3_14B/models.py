import torch
import torch.nn as nn
import torch.nn.functional as F

#class LLMForSequenceClassification(nn.Module):
#    def __init__(self, base_model, num_labels):
#        super().__init__()
#        self.base_model = base_model
#        self.dropout = nn.Dropout(0.1)
#        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_labels)
#
#    def forward(self, input_ids, attention_mask=None, labels=None):
#        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
#        pooled_output = outputs.last_hidden_state[:, -1, :]
#        pooled_output = self.dropout(pooled_output)
#        logits = self.classifier(pooled_output)
#
#        loss = None
#        if labels is not None:
#            loss_fct = nn.CrossEntropyLoss()
#            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
#
#        return type("Output", (), {"loss": loss, "logits": logits})()

class LLMForSequenceClassification(nn.Module):
    def __init__(self, base_model, num_labels, pooling_strategy='last_token', 
                 use_adapter=False, adapter_dim=5120, dropout_rate=0.1):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.pooling_strategy = pooling_strategy
        self.dropout = nn.Dropout(dropout_rate)
        
        hidden_size = self.base_model.config.hidden_size
        
        # Optional adapter layer for better fine-tuning
        if use_adapter:
            self.adapter = nn.Sequential(
                nn.Linear(hidden_size, adapter_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(adapter_dim, hidden_size)
            )
        else:
            self.adapter = None
            
        # Classification head with optional intermediate layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights with small random values"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def get_pooled_representation(self, hidden_states, attention_mask):
        """Get pooled representation based on strategy"""
        if self.pooling_strategy == 'last_token':
            # Get the last non-padding token for each sequence
            batch_size = hidden_states.size(0)
            sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            return hidden_states[torch.arange(batch_size), sequence_lengths]
            
        elif self.pooling_strategy == 'mean':
            # Mean pooling over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
            
        elif self.pooling_strategy == 'max':
            # Max pooling over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states = hidden_states.masked_fill(~mask_expanded.bool(), -1e9)
            return torch.max(hidden_states, dim=1)[0]
            
        elif self.pooling_strategy == 'cls_like':
            # Use first token (similar to BERT's [CLS])
            return hidden_states[:, 0, :]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.base_model.config.pad_token_id).long()
        
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False  # Disable cache for training
        )
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state
        
        # Apply adapter if present
        if self.adapter is not None:
            hidden_states = hidden_states + self.adapter(hidden_states)
        
        # Pool the representations
        pooled_output = self.get_pooled_representation(hidden_states, attention_mask)
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # Classification
                loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing
                loss = loss_fct(logits, labels)
        
        return type("ClassificationOutput", (), {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states if kwargs.get("output_hidden_states") else None
        })()