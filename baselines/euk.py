import torch
import torch.nn as nn

from thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate
from thirdparty.repdistiller.helper.loops import train_vanilla

import copy


def cfk_unlearn(model_cfk, r_loader, model_name):
    """CF-k (Catastrophic Forgetting-k) baseline: freezes every parameter
    except the last conv/residual block, then fine-tunes only that block on
    the remain set. The idea is that forgetting is concentrated in the last
    block, so retraining just it approximates full retraining cheaply."""
    lr_decay_epochs = [10,15,20]
    cfk_lr = 0.01
    cfk_epochs = 10

    for param in model_cfk.parameters():
        param.requires_grad_(False)

    if model_name == 'allcnn':
        layers = [9]
        for k in layers:
            for param in model_cfk.features[k].parameters():
                param.requires_grad_(True)

    elif model_name in ("resnet", "resnet50"):
        for param in model_cfk.resnet_base.layer4.parameters():
            param.requires_grad_(True)

    elif model_name == 'vit':
        # Last transformer block only - the classifier head stays frozen,
        # mirroring allcnn/resnet's "unfreeze only the last representational
        # block" convention above.
        for param in model_cfk.vit.blocks[-1].parameters():
            param.requires_grad_(True)

    else:
        raise NotImplementedError


    fk_fientune(model_cfk, r_loader, lr_decay_epochs, epochs=cfk_epochs, quiet=False, lr=cfk_lr)

    return model_cfk




def euk_unlearn(model, r_loader, model_name):
    """EU-k (Exact Unlearning-k) baseline: like CF-k, but first resets the
    last block's weights back to model_initial before fine-tuning it on the
    remain set - intended to more aggressively remove whatever that block
    learned before retraining it fresh."""
    lr_decay_epochs = [10,15,20]
    euk_lr = 0.01
    euk_epochs = 10
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

    elif model_name in ("resnet", "resnet50"):
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