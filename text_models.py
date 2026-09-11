"""TextTransformer: a DistilBERT-based text classifier exposing the same
`forward`/`get_embedding`/`forward_with_features` hook API as `models.py`'s
AllCNN/CustomResNet/ViT, so gear.py's contrastive/entanglement losses and
every baseline work on it unmodified.

Deliberately kept in its own module, separate from models.py, and imported
lazily (only inside the `elif model_name == 'distilbert':` branches that
actually need it, e.g. trainer.py, baseline_utils.py) rather than at the top
of any shared file - `transformers` is a new dependency this pivot adds, and
nothing about existing vision experiments should break (import-time or
otherwise) on a machine that hasn't installed it yet.

Input convention (why this file's forward()/get_embedding() take a single
tensor `x`, not separate input_ids/attention_mask arguments): every existing
piece of shared code in this repo - gear.py's loss functions, every
baseline, trainer.py's train/eval loops, class_hierarchy.py,
ain_metric.py - calls `model(batch_x)` on a single tensor pulled straight
off a DataLoader and moved via `.to(device)`. Rather than touching all of
those call sites, text_data.py's dataset packs input_ids and attention_mask
into one stacked tensor of shape [2, seq_len] per example (so a batch is
[B, 2, seq_len]); this class unpacks that stack internally. To every other
function in this codebase, a batch of text still just looks like "a tensor
x, model(x) returns logits" - identical to a batch of images.
"""
from torch import nn


class TextTransformer(nn.Module):
    """A pretrained DistilBERT encoder with a replaced classification head.

    get_embedding returns the CLS-token pooled representation (DistilBERT
    has no separate pooler layer like BERT does - hidden_state[:, 0, :] is
    the standard, widely-used stand-in), the same role ViT's CLS token
    plays via forward_head(..., pre_logits=True)."""

    def __init__(self, num_classes, model_name='distilbert-base-uncased'):
        super(TextTransformer, self).__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def _unpack(self, x):
        """x: [B, 2, seq_len] (input_ids and attention_mask stacked at dim 1,
        produced by text_data.py) -> (input_ids, attention_mask), both
        [B, seq_len] long tensors."""
        input_ids = x[:, 0, :].long()
        attention_mask = x[:, 1, :].long()
        return input_ids, attention_mask

    def forward(self, x):
        input_ids, attention_mask = self._unpack(x)
        hidden_state = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = hidden_state[:, 0, :]
        return self.classifier(pooled)

    def get_embedding(self, x):
        """Returns the CLS-token pooled representation before the classifier head."""
        input_ids, attention_mask = self._unpack(x)
        hidden_state = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return hidden_state[:, 0, :]

    def forward_with_features(self, x, capture_layers=None):
        """Runs the forward pass with forward hooks on the named transformer
        blocks (self.encoder.transformer.layer[i], DistilBERT's own naming -
        6 blocks for distilbert-base-uncased) to capture their intermediate
        activations, addressed by integer block index - same convention as
        models.ViT.forward_with_features. Only the CLS token (index 0 of the
        token sequence) is kept per captured block, for the same reason ViT
        does: keeps captured activations a plain [B, D] vector, not a full
        [B, seq_len, D] token sequence. Hooks are always removed afterward so
        repeated calls don't leak/accumulate."""
        blocks = self.encoder.transformer.layer
        num_blocks = len(blocks)
        if capture_layers is None:
            capture_layers = [num_blocks - 1]
        activations = {}
        hooks = []
        for layer_idx in capture_layers:
            if not (0 <= layer_idx < num_blocks):
                raise ValueError(
                    f"Block index {layer_idx} out of range. Valid: 0-{num_blocks - 1}"
                )
            def make_hook(idx):
                def hook(module, inp, out):
                    # DistilBERT transformer-block forward returns a tuple;
                    # the hidden-state tensor is always the first element.
                    hidden = out[0] if isinstance(out, tuple) else out
                    activations[idx] = hidden[:, 0, :]
                return hook
            hooks.append(blocks[layer_idx].register_forward_hook(make_hook(layer_idx)))

        logits = self.forward(x)

        for h in hooks:
            h.remove()
        return logits, activations
