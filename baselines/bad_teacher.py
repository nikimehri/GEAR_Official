"""Bad Teacher baseline (Chundawat et al., AAAI 2023, "Can Bad Teaching
Induce Forgetting? Unlearning in Deep Networks using an Incompetent
Teacher", https://github.com/vikram2000b/bad-teaching-unlearning).

Adapted from the reference repo (MIT licensed), specifically unlearn.py's
UnlearnerLoss/blindspot_unlearner. Core idea: a dual-teacher knowledge-
distillation scheme. The STUDENT (a copy of the model being unlearned) is
trained to imitate two frozen teachers depending on each sample's forget/
retain membership:
  - On RETAIN samples: imitate the "competent" teacher - the original,
    already-trained model being unlearned itself (frozen).
  - On FORGET samples: imitate an "incompetent" teacher - a FRESH, randomly
    initialized model of the same architecture, which has never been
    trained on anything and so outputs close to a random/noisy
    distribution.
The loss is one KL divergence per batch, between the student's temperature-
scaled softmax and a per-sample BLEND of the two teachers' temperature-
scaled softmaxes, blended by each sample's binary forget/retain membership
(exactly the reference's UnlearnerLoss - a convex combination, not two
separate loss terms).

One real adaptation this repo needs that the reference doesn't: this
repo's pretrained architectures (CustomResNet, VGG, ViT) always load
ImageNet-pretrained weights inside their own __init__ (see models.py) -
there's no "skip pretraining" constructor argument. The reference's own
examples specifically construct the incompetent teacher with
pretrained=False (a genuinely untrained, random network - see their
CIFARSuper20_Rocket_Unlearn.ipynb), which is the whole point of
"incompetent." To get an equally genuinely-untrained teacher regardless of
architecture, reinitialize_weights() below constructs the right
architecture via the normal load_model() path (which may load pretrained
weights) and then calls .reset_parameters() on every submodule that has
one - the standard, architecture-agnostic way to scramble an already-
constructed model back to a fresh random initialization. AllCNN never
loads pretrained weights in the first place, so this is a no-op correction
for that architecture and a necessary one for every pretrained backbone.

Interleaving: the reference builds one combined (forget+retain) Dataset
with a per-sample membership label and shuffles it as a single stream, so
a typical mini-batch mixes both. This implementation achieves the same
mixing without a custom Dataset wrapper by pairing one forget-loader batch
with one retain-loader batch per step (retain cycled via gear.inf_generator,
since it's usually much larger than forget) and concatenating them into one
combined batch before computing the single KL loss - the same established
pattern baselines/cu.py already uses for forget/retain batch pairing.
"""
import copy
import sys
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gear import inf_generator
from baseline_utils import load_model


def reinitialize_weights(model):
    """Resets every submodule with learnable parameters (Conv/Linear/
    BatchNorm/etc.) back to a fresh random initialization, in place -
    scrambles away any pretrained weights load_model's constructors may
    have loaded, regardless of architecture. Returns model for convenience."""
    for module in model.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
    return model


def _kl_distill_loss(student_logits, full_teacher_logits, incompetent_teacher_logits,
                     forget_membership, temperature):
    """The reference's UnlearnerLoss exactly: one KL divergence between the
    student's temperature-scaled softmax and a per-sample convex blend of
    the two teachers' temperature-scaled softmaxes, blended by
    forget_membership (1=forget -> incompetent teacher, 0=retain ->
    competent/full teacher)."""
    full_probs = F.softmax(full_teacher_logits / temperature, dim=1)
    incompetent_probs = F.softmax(incompetent_teacher_logits / temperature, dim=1)
    membership = forget_membership.unsqueeze(1)
    target_probs = membership * incompetent_probs + (1 - membership) * full_probs
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, target_probs, reduction='batchmean')


def bad_teacher_unlearn(model, train_forget_loader, train_remain_loader, model_type, num_classes,
                        data_name, device, epochs=1, lr=1e-4, temperature=1.0):
    """Returns a new, unlearned copy of model (model itself is untouched).

    model_type/num_classes/data_name are needed to construct the
    incompetent teacher via load_model() with the exact same architecture/
    output shape as model, before reinitialize_weights() scrambles it back
    to a fresh random init."""
    student = copy.deepcopy(model).to(device)
    student.train()

    competent_teacher = copy.deepcopy(model).to(device)
    competent_teacher.eval()
    for p in competent_teacher.parameters():
        p.requires_grad_(False)

    incompetent_teacher = load_model(model_type, num_classes=num_classes, data_name=data_name).to(device)
    reinitialize_weights(incompetent_teacher)
    incompetent_teacher.eval()
    for p in incompetent_teacher.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    retain_gen = inf_generator(train_remain_loader)

    for _epoch in range(epochs):
        for x_f, _y_f in train_forget_loader:
            x_r, _y_r = next(retain_gen)
            x_f, x_r = x_f.to(device), x_r.to(device)

            x = torch.cat([x_f, x_r], dim=0)
            forget_membership = torch.cat([
                torch.ones(x_f.size(0), device=device),
                torch.zeros(x_r.size(0), device=device),
            ])

            with torch.no_grad():
                full_logits = competent_teacher(x)
                incompetent_logits = incompetent_teacher(x)
            student_logits = student(x)

            loss = _kl_distill_loss(student_logits, full_logits, incompetent_logits,
                                    forget_membership, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    student.eval()
    return student
