from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
import sys
import os

# from mia_metric import *
from thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate
from thirdparty.repdistiller.distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from thirdparty.repdistiller.distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss

from thirdparty.repdistiller.helper.loops import train_distill, train_distill_hide, train_distill_linear, train_vanilla, train_negrad, train_bcu, train_bcu_distill, validate
from thirdparty.repdistiller.helper.pretrain import init

import copy
import time


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gear import (
    RESNET_ALL_LAYERS,
    get_intermediate_features,
    get_intermediate_features_multilayer,
    prepare_features,
    retain_alignment_loss,
    retain_forget_loss,
    forget_forget_loss,
    compute_retain_centroids,
    compute_entanglement_scores,
    _get_primary_layer,
    _resolve_deepest_layer,
)


def scrub_met(teacher, student, remain_loader, forget_loader,model_name,dataset,seed=2022):
    optim_name  = 'sgd'
    gamma = 1
    alpha = 0.5
    beta = 0
    smoothing = 0.5
    msteps = 3
    clip = 0.2
    sstart = 10
    kd_T = 4
    distill = 'kd'

    sgda_epochs = 5
    sgda_learning_rate = 0.0005
    lr_decay_epochs = [3,5,9]
    lr_decay_rate = 0.1
    sgda_weight_decay = 5e-4
    sgda_momentum = 0.9

    model_t = teacher
    model_s = student


    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(kd_T)
    criterion_kd = DistillKL(kd_T)


    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    
    criterion_list.append(criterion_div)    
    criterion_list.append(criterion_kd)     

    # optimizer
    if optim_name  == "sgd":
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=sgda_learning_rate,
                              momentum=sgda_momentum,
                              weight_decay=sgda_weight_decay)

    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True


    t1 = time.time()
    acc_rs = []
    acc_fs = []
    acc_vs = []
    acc_fvs = []
    

    
    scrub_name = "checkpoints/scrub_{}_{}_seed{}_step".format(model_name, dataset, seed)
    for epoch in range(1, sgda_epochs + 1):

        lr = sgda_adjust_learning_rate(epoch, optimizer,lr_decay_epochs)

        acc_r, acc5_r, loss_r = validate(remain_loader, model_s, criterion_cls,  True)
        acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls,  True)

        acc_rs.append(100-acc_r.item())
        acc_fs.append(100-acc_f.item())


        maximize_loss = 0
        if epoch <= msteps:
            maximize_loss = train_distill(epoch, forget_loader, module_list, None, criterion_list, optimizer,  "maximize")
        
        
        train_acc, train_loss = train_distill(epoch, remain_loader, module_list, None, criterion_list, optimizer,  "minimize",)
        
        torch.save(model_s.state_dict(), scrub_name+str(epoch)+".pt")


        print ("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss, train_loss, train_acc))
    
    t2 = time.time()
    print (t2-t1)

    acc_r, acc5_r, loss_r = validate(remain_loader, model_s, criterion_cls,  True)
    acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls,  True)

    acc_rs.append(100-acc_r.item())
    acc_fs.append(100-acc_f.item())

    
    try:
        selected_idx, _ = min(enumerate(acc_fs), key=lambda x: abs(x[1]-acc_fvs[-1]))
    except:
        selected_idx = len(acc_fs) - 1

    print ("the selected index is {}".format(selected_idx))
    selected_model = "checkpoints/scrub_{}_{}_seed{}_step{}.pt".format(model_name, dataset, seed, int(selected_idx))
    model_s_final = copy.deepcopy(model_s)
    model_s.load_state_dict(torch.load(selected_model))


    return model_s, model_s_final


def scrub_unlearn(teacher, student, remain_loader, forget_loader, model_name, dataset, seed=2022,
                  sgda_epochs=5,
                  # CL+ES args — all default to off so existing callers are unaffected
                  feature_contrastive=False,
                  use_entanglement_weighting=False,
                  retain_forget_weight=2.0,
                  forget_forget_weight=3.0,
                  feature_align_weight=0.0,
                  gamma_rep=1.0,
                  remain_reg=3.5,
                  centroid_refresh_interval=None,
                  beta_ce=0.0,
                  target_layer='layer4',
                  num_classes=None,
                  target_forget_acc=None,
                  ):
    """SCRUB unlearning with optional CL+ES contrastive regularisation.

    When feature_contrastive=False (default) this function is identical to
    scrub_met in every observable way.

    When feature_contrastive=True, after each epoch's minimize pass a single
    CL+ES pass is run over paired (forget, retain) batches before the
    checkpoint is saved.  Feature hooks, centroid refresh logic, entanglement
    scoring, and loss terms are the same as in gear.gear.

    Args:
        target_layer: passed through from --target_layer CLI arg (same arg
                      that controls the PSG unlearning loop layer).
        target_forget_acc: the retrain (gold-standard) model's forget-set
                      accuracy, as a fraction in [0, 1]. When given, the
                      "best" checkpoint is the epoch whose forget accuracy
                      comes closest to this value — matching the published
                      SCRUB selection heuristic (forgetting to the same
                      degree a retrained-from-scratch model would, rather
                      than forgetting as much as possible, which can be a
                      sign of collateral damage to the retain set). When
                      not given (no retrain model was available to the
                      caller), falls back to the last epoch's checkpoint.
    """
    optim_name = 'sgd'
    gamma = 1
    alpha = 0.5
    beta = 0
    smoothing = 0.5
    msteps = 3
    clip = 0.2
    sstart = 10
    kd_T = 4
    distill = 'kd'

    # sgda_epochs comes from the function parameter (default 5)
    sgda_learning_rate = 0.0005
    lr_decay_epochs = [3, 5, 9]
    lr_decay_rate = 0.1
    sgda_weight_decay = 5e-4
    sgda_momentum = 0.9

    model_t = teacher
    model_s = student

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(kd_T)
    criterion_kd = DistillKL(kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)
    criterion_list.append(criterion_div)
    criterion_list.append(criterion_kd)

    # SCRUB optimizer (unchanged from scrub_met)
    if optim_name == "sgd":
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=sgda_learning_rate,
                              momentum=sgda_momentum,
                              weight_decay=sgda_weight_decay)

    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True


    device = next(model_s.parameters()).device

    # Separate optimizer for the CL+ES step — same lr/momentum as the PSG loop.
    # Kept separate so it doesn't interfere with SCRUB's own optimizer state.
    cl_es_optimizer = (
        torch.optim.SGD(model_s.parameters(), lr=0.0001, momentum=0.9)
        if feature_contrastive else None
    )

    # Resolve the primary layer for centroid / entanglement computation.
    # Multi-layer + entanglement: dynamic lookup (DataParallel-safe).
    # All other cases: existing _get_primary_layer behaviour.
    if feature_contrastive and use_entanglement_weighting and target_layer == 'all':
        _primary_layer = _resolve_deepest_layer(model_s, RESNET_ALL_LAYERS)
    else:
        _primary_layer = _get_primary_layer(target_layer)

    # Default refresh interval to one epoch's worth of CL+ES steps
    if centroid_refresh_interval is None:
        centroid_refresh_interval = len(forget_loader)

    centroid_cache = None                                   # plain tensor buffer
    _num_classes = num_classes                              # inferred lazily if None
    _ce_per_sample = nn.CrossEntropyLoss(reduction='none')  # for targeted CE
    cl_es_step = 0                                          # persists across epochs

    if feature_contrastive:
        print(f'[scrub_unlearn] CL+ES enabled | target_layer={target_layer} | '
              f'use_entanglement_weighting={use_entanglement_weighting} | '
              f'centroid_refresh_interval={centroid_refresh_interval} | '
              f'primary_layer={_primary_layer}')
    # -------------------------------------------------------------------------

    t1 = time.time()
    acc_rs = []
    acc_fs = []

    scrub_name = "checkpoints/scrub_{}_{}_seed{}_step".format(model_name, dataset, seed)
    os.makedirs(os.path.dirname(scrub_name), exist_ok=True)

    for epoch in range(1, sgda_epochs + 1):

        lr = sgda_adjust_learning_rate(epoch, optimizer, lr_decay_epochs)

        acc_r, acc5_r, loss_r = validate(remain_loader, model_s, criterion_cls, True)
        acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls, True)

        acc_rs.append(100 - acc_r.item())
        acc_fs.append(100 - acc_f.item())

        maximize_loss = 0
        if epoch <= msteps:
            maximize_loss = train_distill(epoch, forget_loader, module_list, None, criterion_list, optimizer, "maximize")

        train_acc, train_loss = train_distill(epoch, remain_loader, module_list, None, criterion_list, optimizer, "minimize")


        if feature_contrastive:
            model_s.train()
            cl_es_epoch_loss = 0.0

            for (x_f, y_f), (x_r, y_r) in zip(forget_loader, remain_loader):
                x_f = x_f.to(device)
                y_f = y_f.to(device)
                x_r = x_r.to(device)
                y_r = y_r.to(device)

                # -- Entanglement: centroid refresh (same logic as PSG loop) --
                e_scores = None
                _ff_for_ent = None
                _rf_for_tce = None

                if use_entanglement_weighting:
                    # Lazy num_classes inference
                    if _num_classes is None:
                        with torch.no_grad():
                            _num_classes = model_s(x_r[:1]).size(1)

                    # Same refresh condition as gear.gear
                    if centroid_cache is None or (cl_es_step > 0 and cl_es_step % centroid_refresh_interval == 0):
                        centroid_cache = compute_retain_centroids(
                            model_s, remain_loader, _primary_layer, _num_classes, device
                        )
                        # compute_retain_centroids restores training mode internally
                        model_s.train()

                # -- Feature extraction and contrastive losses ----------------
                if target_layer == 'all':
                    layers = RESNET_ALL_LAYERS
                    forget_feats_dict = get_intermediate_features_multilayer(model_s, x_f, layers)
                    retain_feats_dict = get_intermediate_features_multilayer(model_s, x_r, layers)
                    with torch.no_grad():
                        retain_ref_feats_dict = get_intermediate_features_multilayer(model_t, x_r, layers)

                    # Reuse deepest-layer features for e_scores — no extra forward pass.
                    # Mirrors the multi-layer path in gear.gear.
                    if use_entanglement_weighting and centroid_cache is not None:
                        with torch.no_grad():
                            _ff_for_ent = prepare_features(forget_feats_dict[_primary_layer].detach())
                            _rf_for_tce = prepare_features(retain_feats_dict[_primary_layer].detach())
                            e_scores = compute_entanglement_scores(_ff_for_ent, centroid_cache)

                    align_loss = torch.tensor(0.0, device=device)
                    rf_loss = torch.tensor(0.0, device=device)
                    ff_loss = torch.tensor(0.0, device=device)
                    for layer in layers:
                        f_f = prepare_features(forget_feats_dict[layer])
                        r_f = prepare_features(retain_feats_dict[layer])
                        r_ref_f = prepare_features(retain_ref_feats_dict[layer])
                        align_loss = align_loss + retain_alignment_loss(r_f, r_ref_f)
                        rf_loss = rf_loss + retain_forget_loss(r_f, f_f, entanglement_scores=e_scores)
                        ff_loss = ff_loss + forget_forget_loss(f_f)

                else:
                    # Single-layer mode — mirrors the single-layer path in PSG loop.
                    forget_feats = get_intermediate_features(model_s, x_f, target_layer)
                    retain_feats = get_intermediate_features(model_s, x_r, target_layer)
                    with torch.no_grad():
                        retain_ref_feats = get_intermediate_features(model_t, x_r, target_layer)

                    forget_feats = prepare_features(forget_feats)
                    retain_feats = prepare_features(retain_feats)
                    retain_ref_feats = prepare_features(retain_ref_feats)

                    # Detach for e_scores; forget_feats/retain_feats keep grad for losses.
                    if use_entanglement_weighting and centroid_cache is not None:
                        with torch.no_grad():
                            _ff_for_ent = forget_feats.detach()
                            _rf_for_tce = retain_feats.detach()
                            e_scores = compute_entanglement_scores(_ff_for_ent, centroid_cache)

                    align_loss = retain_alignment_loss(retain_feats, retain_ref_feats)
                    rf_loss = retain_forget_loss(retain_feats, forget_feats, entanglement_scores=e_scores)
                    ff_loss = forget_forget_loss(forget_feats)

                contrastive_loss = gamma_rep * (
                    feature_align_weight * align_loss +
                    retain_forget_weight * rf_loss +
                    forget_forget_weight * ff_loss
                )

                # -- Targeted CE (only when entanglement weighting is on) ------
                targeted_ce_val = torch.tensor(0.0, device=device)
                if use_entanglement_weighting and beta_ce > 0.0 and e_scores is not None:
                    remain_logits = model_s(x_r)
                    with torch.no_grad():
                        cross_sims = torch.mm(_rf_for_tce, _ff_for_ent.t())   # [B_r, B_f]
                        w_r = (cross_sims * e_scores.unsqueeze(0)).max(dim=1).values
                        w_r = w_r.clamp(min=0.0)
                        w_r = w_r / (w_r.sum() + 1e-8) * w_r.size(0)
                    per_sample_ce = _ce_per_sample(remain_logits, y_r)
                    targeted_ce_val = (w_r * per_sample_ce).mean()

                cl_es_loss = contrastive_loss + beta_ce * targeted_ce_val

                cl_es_optimizer.zero_grad()
                cl_es_loss.backward()
                cl_es_optimizer.step()

                cl_es_epoch_loss += cl_es_loss.item()
                cl_es_step += 1

            print(f"epoch {epoch} | CL+ES loss: {cl_es_epoch_loss:.4f}")
        # -----------------------------------------------------------------

        # Checkpoint saved after both SCRUB and CL+ES updates each epoch
        torch.save(model_s.state_dict(), scrub_name + str(epoch) + ".pt")

        print("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(
            maximize_loss, train_loss, train_acc))

    t2 = time.time()
    print(t2 - t1)

    acc_r, acc5_r, loss_r = validate(remain_loader, model_s, criterion_cls, True)
    acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls, True)

    acc_rs.append(100 - acc_r.item())
    acc_fs.append(100 - acc_f.item())

    # Select the checkpoint whose forget-error is closest to the retrain
    # model's forget-error, if one was supplied; otherwise use the last
    # epoch's checkpoint (the same behavior scrub_met falls back to, but
    # explicit here rather than via an always-failing lookup).
    # acc_fs[0] is measured before epoch 1 trains, so it has no matching
    # checkpoint file (only steps 1..sgda_epochs were saved) - excluded here.
    if target_forget_acc is not None:
        target_error = 100 - target_forget_acc * 100
        candidates = list(enumerate(acc_fs))[1:]
        selected_idx, _ = min(candidates, key=lambda x: abs(x[1] - target_error))
    else:
        selected_idx = len(acc_fs) - 1

    print("the selected index is {}".format(selected_idx))
    selected_model = "checkpoints/scrub_{}_{}_seed{}_step{}.pt".format(
        model_name, dataset, seed, int(selected_idx))
    model_s_final = copy.deepcopy(model_s)
    model_s.load_state_dict(torch.load(selected_model))

    return model_s, model_s_final

