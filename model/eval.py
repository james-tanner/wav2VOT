import torch
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from jiwer import wer

def convert_to_sequence(seq, blank_id=4):
    '''
    convert framewise labels into single sequence
    (removing adjacent duplicates)
    '''
    sequence = []
    prev = None
    for token in seq:
        if token != blank_id and token != prev:
            sequence.append(str(token))
        prev = token
    return sequence

def compute_sequence_wer(preds, labels):
    # Ensure 2D shape (B, T)
    if preds.ndim == 1:
        preds = preds[np.newaxis, :]
        labels = labels[np.newaxis, :]

    wers = []
    for p_seq, l_seq in zip(preds, labels):
        p_clean = convert_to_sequence(p_seq[p_seq != -100])
        l_clean = convert_to_sequence(l_seq[l_seq != -100])

        p_str = " ".join(map(str, p_clean))
        l_str = " ".join(map(str, l_clean))

        wers.append(wer(l_str, p_str))

    return float(np.mean(wers))

def compute_eval_metrics(pred):
    ## keep framewise logits and ignore length
    logits, _ = pred.predictions
    logits = torch.tensor(logits)

    ## reshape predictions to framewise class logits
    batch_size, _, num_classes = logits.shape
    logits = logits.view(batch_size, -1, num_classes)

    ## Get predicted classes (highest logits)
    preds = logits.argmax(dim=-1)

    ## convert reference labels to tensor
    labels = torch.tensor(pred.label_ids)

    ## Interpolate labels to match sequence length (if needed)
    if labels.shape[1] != logits.shape[1]:
        labels = labels.unsqueeze(1).float()  # (batch_size, 1, original_length)
        labels = F.interpolate(labels, size=logits.shape[1], mode="nearest").squeeze(1).long()
    assert preds.shape == labels.shape, f"Shape mismatch: {preds.shape} vs {labels.shape}"

    ## Mask out padding (-100)
    active_indices = labels != -100

    ## Flatten to 1D for scikit-learn compatibility
    preds = preds[active_indices].cpu().numpy()
    labels = labels[active_indices].cpu().numpy()

    ## calculate metrics
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average = "macro")
    avg_wer = compute_sequence_wer(preds, labels)

    return {
        "framewise_accuracy": round(accuracy, 3),
        "framewise_f1" : round(f1, 3),
        "sequence_wer": round(avg_wer, 3),
    }
