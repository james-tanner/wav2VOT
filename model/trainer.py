import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

class Wav2VOTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        ## run model inference
        outputs = model(**inputs)

        ## extract model predictions
        logits = outputs["logits"]
        logits_lengths = outputs['logit_lengths']
        log_probs = F.log_softmax(logits, dim=-1)

        ## reference information
        labels = inputs["labels"]
        sequence_targets = inputs["seq_targets"]
        target_lengths = inputs["target_lengths"]

        loss = None
        if labels is not None:
            ## Interpolate labels to match frames
            labels = labels.unsqueeze(1).float()
            labels = F.interpolate(labels, size=logits.shape[1], mode="nearest")
            labels = labels.squeeze(1).long()

            ## calculate losses
            loss_fct = nn.CrossEntropyLoss(ignore_index = -100, label_smoothing = 0.1)
            framewise_loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

            ## get CTC loss
            if sequence_targets is not None:
                log_probs_ctc = log_probs.permute(1, 0, 2)
                ctc_loss_fn = nn.CTCLoss(blank=4, zero_infinity=True)
                ctc_loss = ctc_loss_fn(
                    log_probs_ctc,
                    sequence_targets,
                    logits_lengths,
                    target_lengths
                )

                ## scale CTC loss down
                alpha = 0.05
                loss = framewise_loss + ctc_loss * alpha
            else:
                loss = framewise_loss

        ## return losses for logging
        self.log({
            "alpha": alpha,
            "framewise_loss": framewise_loss.item(),
            "ctc_loss": ctc_loss.item(),
        })

        if return_outputs:
            return loss, outputs
        else:
            return loss
