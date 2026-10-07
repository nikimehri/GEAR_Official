import torch
import torch.nn as nn

from thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate
from thirdparty.repdistiller.helper.loops import train_vanilla

import copy

# Single source of truth for which architectures cfk_unlearn/euk_unlearn
# support - both functions' model_name branches below define this set in
# practice (anything not in it hits their `else: raise NotImplementedError`).
# Exposed here so callers (baseline_main.py's dispatch) can check support
# BEFORE calling in and skip cleanly, the same way COUN_VALID_PAIRINGS lets
# coun's dispatch branch skip cleanly for a config it doesn't support -
# instead of letting an uncaught NotImplementedError crash the whole process
# partway through a multi-method --method run. Unlike COUN_VALID_PAIRINGS,
# this isn't keyed by dataset: cfk/euk's branches only ever check
# model_name, never data_name, so a flat set is the accurate representation.
CFK_EUK_SUPPORTED_MODELS = {'allcnn', 'resnet', 'resnet50', 'resnet18', 'vit', 'vgg16', 'vgg16_bn', 'distilbert'}


def cfk_unlearn(model_cfk, r_loader, model_name, cfk_lr=0.01, cfk_epochs=10, lr_decay_epochs=(10, 15, 20)):
    """CF-k (Catastrophic Forgetting-k) baseline: freezes every parameter
    except the last conv/residual block, then fine-tunes only that block on
    the remain set. The idea is that forgetting is concentrated in the last
    block, so retraining just it approximates full retraining cheaply.

    cfk_lr/cfk_epochs/lr_decay_epochs were hardcoded constants until a
    hyperparameter search needed them exposed (see baselines/hpo_search.py)
    - the defaults here are exactly the old hardcoded values, so every
    existing caller is unaffected."""
    lr_decay_epochs = list(lr_decay_epochs)

    for param in model_cfk.parameters():
        param.requires_grad_(False)

    if model_name == 'allcnn':
        layers = [9]
        for k in layers:
            for param in model_cfk.features[k].parameters():
                param.requires_grad_(True)

    elif model_name in ("resnet", "resnet50", "resnet18"):
        # resnet18 shares the same named-stage structure as resnet50
        # (models.CustomResNet's arch parameter only changes channel widths),
        # so this branch needs no new logic, just the extra name.
        for param in model_cfk.resnet_base.layer4.parameters():
            param.requires_grad_(True)

    elif model_name == 'vit':
        # Last transformer block only - the classifier head stays frozen,
        # mirroring allcnn/resnet's "unfreeze only the last representational
        # block" convention above.
        for param in model_cfk.vit.blocks[-1].parameters():
            param.requires_grad_(True)

    elif model_name == 'distilbert':
        # Same "last representational block only" convention, DistilBERT's
        # naming (self.encoder.transformer.layer, 6 blocks).
        for param in model_cfk.encoder.transformer.layer[-1].parameters():
            param.requires_grad_(True)

    elif model_name in ('vgg16', 'vgg16_bn'):
        # VGG has no ResNet-style stages or ViT-style block list - the
        # closest analogue to "last representational block" is fc7's own
        # Linear(4096, 4096) (classifier[3]; ReLU/Dropout at [4]/[5] have no
        # parameters), the same layer models.VGG.get_embedding/
        # forward_with_features already treat as this architecture's
        # penultimate representation. The final Linear(4096, num_classes) at
        # classifier[6] stays frozen, matching every other branch above.
        for param in model_cfk.vgg.classifier[3].parameters():
            param.requires_grad_(True)

    else:
        raise NotImplementedError


    fk_fientune(model_cfk, r_loader, lr_decay_epochs, epochs=cfk_epochs, quiet=False, lr=cfk_lr)

    return model_cfk




def euk_unlearn(model, r_loader, model_name, euk_lr=0.01, euk_epochs=10, lr_decay_epochs=(10, 15, 20)):
    """EU-k (Exact Unlearning-k) baseline: like CF-k, but first resets the
    last block's weights back to model_initial before fine-tuning it on the
    remain set - intended to more aggressively remove whatever that block
    learned before retraining it fresh.

    euk_lr/euk_epochs/lr_decay_epochs were hardcoded constants until a
    hyperparameter search needed them exposed (see baselines/hpo_search.py)
    - the defaults here are exactly the old hardcoded values, so every
    existing caller is unaffected."""
    lr_decay_epochs = list(lr_decay_epochs)
    model_initial = model
    model_euk = copy.deepcopy(model)

    for param in model_euk.parameters():
        param.requires_grad_(False)

    if model_name == 'allcnn':
        layers = [9]

        with torch.no_grad():
            for k in layers:
                for i in range(0,3):
                    try:
                        model_euk.features[k][i].weight.copy_(model_initial.features[k][i].weight)
                    except:
                        print ("block {}, layer {} does not have weights".format(k,i))
                    try:
                        model_euk.features[k][i].bias.copy_(model_initial.features[k][i].bias)
                    except:
                        print ("block {}, layer {} does not have bias".format(k,i))
            model_euk.classifier[0].weight.copy_(model_initial.classifier[0].weight)
            model_euk.classifier[0].bias.copy_(model_initial.classifier[0].bias)

        for k in layers:
            for param in model_euk.features[k].parameters():
                param.requires_grad_(True)

    elif model_name in ("resnet", "resnet50", "resnet18"):
        # resnet18's layer4 has exactly 2 BasicBlocks (vs. resnet50's 3
        # Bottleneck blocks) - range(0,2) below covers all of resnet18's
        # layer4, same attribute names (bn1/conv1/bn2/conv2/downsample) on
        # both block types.
        with torch.no_grad():
            for i in range(0,2):
                try:
                    model_euk.resnet_base.layer4[i].bn1.weight.copy_(model_initial.resnet_base.layer4[i].bn1.weight)
                except:
                    print ("block 4, layer {} does not have weight".format(i))
                try:
                    model_euk.resnet_base.layer4[i].bn1.bias.copy_(model_initial.resnet_base.layer4[i].bn1.bias)
                except:
                    print ("block 4, layer {} does not have bias".format(i))
                try:
                    model_euk.resnet_base.layer4[i].conv1.weight.copy_(model_initial.resnet_base.layer4[i].conv1.weight)
                except:
                    print ("block 4, layer {} does not have weight".format(i))
                try:
                    model_euk.resnet_base.layer4[i].conv1.bias.copy_(model_initial.resnet_base.layer4[i].conv1.bias)
                except:
                    print ("block 4, layer {} does not have bias".format(i))

                try:
                    model_euk.resnet_base.layer4[i].bn2.weight.copy_(model_initial.resnet_base.layer4[i].bn2.weight)
                except:
                    print ("block 4, layer {} does not have weight".format(i))
                try:
                    model_euk.resnet_base.layer4[i].bn2.bias.copy_(model_initial.resnet_base.layer4[i].bn2.bias)
                except:
                    print ("block 4, layer {} does not have bias".format(i))
                try:
                    model_euk.resnet_base.layer4[i].conv2.weight.copy_(model_initial.resnet_base.layer4[i].conv2.weight)
                except:
                    print ("block 4, layer {} does not have weight".format(i))
                try:
                    model_euk.resnet_base.layer4[i].conv2.bias.copy_(model_initial.resnet_base.layer4[i].conv2.bias)
                except:
                    print ("block 4, layer {} does not have bias".format(i))

            # model_euk.resnet_base.layer4[0].shortcut[0].weight.copy_(model_initial.resnet_base.layer4[0].shortcut[0].weight)

            if hasattr(model_euk.resnet_base.layer4[0], 'downsample') and model_euk.resnet_base.layer4[0].downsample is not None:
                model_euk.resnet_base.layer4[0].downsample[0].weight.copy_(
                    model_initial.resnet_base.layer4[0].downsample[0].weight
                )


        for param in model_euk.resnet_base.layer4.parameters():
            param.requires_grad_(True)

    elif model_name == 'vit':
        # Reset the last transformer block to its pre-unlearning weights,
        # then unfreeze just that block - ViT blocks are structurally
        # uniform, so a single load_state_dict does what the resnet branch
        # above needs many per-submodule copies for.
        with torch.no_grad():
            model_euk.vit.blocks[-1].load_state_dict(model_initial.vit.blocks[-1].state_dict())

        for param in model_euk.vit.blocks[-1].parameters():
            param.requires_grad_(True)

    elif model_name == 'distilbert':
        # Same reset-then-unfreeze pattern as vit above - DistilBERT's
        # blocks are equally uniform, so one load_state_dict suffices.
        with torch.no_grad():
            model_euk.encoder.transformer.layer[-1].load_state_dict(
                model_initial.encoder.transformer.layer[-1].state_dict()
            )

        for param in model_euk.encoder.transformer.layer[-1].parameters():
            param.requires_grad_(True)

    elif model_name in ('vgg16', 'vgg16_bn'):
        # Same reset-then-unfreeze pattern, same fc7 layer cfk_unlearn
        # unfreezes above (classifier[3], a single Linear(4096, 4096)).
        with torch.no_grad():
            model_euk.vgg.classifier[3].load_state_dict(model_initial.vgg.classifier[3].state_dict())

        for param in model_euk.vgg.classifier[3].parameters():
            param.requires_grad_(True)

    else:
        raise NotImplementedError


    fk_fientune(model_euk, r_loader,lr_decay_epochs, epochs=euk_epochs, quiet=True, lr=euk_lr)
    return model_euk



def fk_fientune(model, data_loader,lr_decay_epochs, lr=0.01, epochs=10, quiet=False):
    """Shared fine-tuning loop used by both cfk_unlearn and euk_unlearn:
    plain SGD with a step-decay schedule."""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
    for epoch in range(epochs):
        sgda_adjust_learning_rate(epoch, optimizer,lr_decay_epochs)
        train_vanilla(epoch, data_loader, model, loss_fn, optimizer)