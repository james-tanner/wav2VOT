import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

class Wav2VOT(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        ## linear projection layer: framewise logits
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_values, labels = None, seq_targets = None, input_lengths = None, target_lengths = None):
        batch_size = input_values.size(0)
        outputs = self.wav2vec2(input_values, output_hidden_states = False)
        hidden_states = outputs.last_hidden_state

        ## get model predictions
        logits = self.classifier(self.dropout(hidden_states))

        ## get the downsampled length of the output
        ## in order to calculate CTC loss
        output_lengths = torch.full((batch_size,), hidden_states.size(1), dtype = torch.long, device = input_values.device)

        return {"logits" : logits, "logit_lengths": output_lengths}
