from transformers import AutoModel, AutoTokenizer, AutoConfig, BertForSequenceClassification
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import os
import json


@dataclass
class ModelOutput:
    loss: torch.Tensor = None
    logits: torch.Tensor = None

class WrapperForTrain(nn.Module):
    def __init__(self, model: BertForSequenceClassification) -> None:
        super().__init__()
        self.model = model
        self.loss_func = nn.MSELoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        labels
    ) -> ModelOutput:
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).logits
        logits = torch.sigmoid(logits)
        loss = None
        if labels is not None:
            loss = self.loss_func(
                logits.view(-1),
                labels.view(-1)
            )
        return ModelOutput(
            loss=loss,
            logits=logits
        )