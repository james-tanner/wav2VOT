import torch
from torch.nn.utils.rnn import pad_sequence

from itertools import groupby

def labels_to_sequence(labels):
    ## convenience function for creating sequence
    return [label for label, _ in groupby(labels) if label != -100]

class Wav2VOTDataCollator:
    def __init__(self, padding_value = 0.0, label_padding_value = -100):
        self.padding_value = padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, features):
        input_values = [f["input_values"].float() for f in features]
        labels = [f["labels"].long() for f in features]

        ## pad audio and framewise labels
        batch_input_values = pad_sequence(input_values, batch_first = True, padding_value = self.padding_value)
        batch_labels = pad_sequence(labels, batch_first = True, padding_value = self.label_padding_value)

        ## pad the target sequence
        seq_targets = [torch.tensor(labels_to_sequence(l.tolist()), dtype = torch.long) for l in labels]
        target_lengths = [len(seq) for seq in seq_targets]
        seq_targets_padded = pad_sequence(seq_targets, batch_first = True, padding_value = 0)

        return {
            "input_values": batch_input_values,
            "labels": batch_labels,
            "seq_targets": seq_targets_padded,
            "target_lengths": torch.tensor(target_lengths, dtype = torch.long),
        }
