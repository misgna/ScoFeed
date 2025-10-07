from transformers import AutoModelForSeq2SeqLM
import torch
import torch.nn as nn

class T5ScoFeed(nn.Module):
    def __init__(self, model_name, num_traits, feature_dim=None):
        super().__init__()
        
        # Load FLAN-T5 backbone (same API as T5)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        hidden_dim = self.model.config.d_model
        
        # If you want to include numeric features (optional)
        if feature_dim is not None:
            input_dim = hidden_dim + feature_dim
        else:
            input_dim = hidden_dim

        # Multi-trait regression head
        self.scoring_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_traits)
        )

    def forward(self, input_ids, attention_mask, labels=None, features=None):
        # Pass through FLAN-T5
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False,
            output_attentions=False,
        )
        
        # Mean-pool encoder hidden states for regression head
        encoder_output = outputs.encoder_last_hidden_state.mean(dim=1)
        
        # Optionally concatenate external features
        if features is not None:
            encoder_output = torch.cat([encoder_output, features], dim=1)
        
        # Predict multi-trait scores
        predicted_scores = self.scoring_head(encoder_output)
        
        return {
            "loss": outputs.loss,               # feedback generation loss
            "logits": outputs.logits,
            "scores": predicted_scores
        }
