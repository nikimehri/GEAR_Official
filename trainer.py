import time
from sklearn.metrics import accuracy_score
import torch
from torch import nn, optim
from tqdm import tqdm
from models import AllCNN, CustomResNet, ViT
import csv
from torchvision import datasets
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os


def load_checkpoint(path, checkpoint_label, device=None):
    """Loads a full pickled model checkpoint, unwrapping DataParallel if
    needed. Raises a clear, specific error instead of the bare torch.load
    call's confusing default errors when the path is missing or unset."""
    if path is None:
        raise ValueError(
            f"Missing checkpoint path for {checkpoint_label}. "
            f"Pass the matching --{checkpoint_label} argument."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint for {checkpoint_label} was not found: {path}"
        )

    model = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    if isinstance(model, nn.DataParallel):
        model = model.module
    if device is not None:
        model = model.to(device)
    return model


def loss_picker(loss, train_loader=None, device='cpu', forget_class=None, num_classes=None):
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cross':
        if train_loader is not None:
            train_labels = []
            for _, labels in train_loader:
                train_labels.extend(labels.numpy())
            unique_classes = np.unique(train_labels)
            raw_weights = compute_class_weight(
                class_weight='balanced', classes=unique_classes, y=train_labels
            )
            # Size the weight tensor to the true num_classes (not just the
            # classes observed in this loader) so weights stay correctly
            # aligned to class index even when a class is entirely absent
            # from train_labels (e.g. the forget class in a remain-only loader).
            weight_tensor = torch.ones(num_classes, dtype=torch.float32)
            for cls, w in zip(unique_classes, raw_weights):
                weight_tensor[cls] = w
            if forget_class is not None:
                weight_tensor[forget_class] = 0.0
            criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        print("Automatically assigning MSE loss function to you...")
        criterion = nn.MSELoss()

    return criterion

def optimizer_picker(optimization, param, lr, momentum=0.):
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        print('Using SGD for optimization')
        optimizer = optim.SGD(param, lr=lr, momentum=momentum, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer '{optimization}', expected 'adam' or 'sgd'")

    return optimizer

def train(model, data_loader, criterion, optimizer, loss_mode, device='cpu'):
    """Runs one epoch of standard supervised training. loss_mode='neg_grad'
    negates the loss before backprop (a crude gradient-ascent option), but
    no current CLI flag selects it - only 'cross'/'mse' are ever used."""
    running_loss = 0
    model.train()
    print(len(data_loader))

    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if len(batch_y.shape) > 1:
            batch_y = batch_y.squeeze()

        optimizer.zero_grad()

        output = model(batch_x)


        if loss_mode == "mse":
            loss = criterion(output, batch_y)
        elif loss_mode == "cross":
            loss = criterion(output, batch_y)
        elif loss_mode == 'neg_grad':
            loss = -criterion(output, batch_y)

        loss.backward()
        optimizer.step()
        running_loss += loss
    return running_loss


def train_save_model(train_loader, val_loader, model_name, optim_name, learning_rate, num_epochs, device, path, dataset=None, relearning=False, unlearned_model=None, data_name=None, forget_class=None):
    start = time.time()
    losses = []
    accuracies = []

    if dataset:
        if isinstance(dataset, datasets.SVHN):
            original_targets = dataset.labels
        elif isinstance(dataset, (datasets.MNIST, datasets.CIFAR10)):
            if isinstance(dataset.targets, torch.Tensor):
                original_targets = dataset.targets.tolist()
            else:
                original_targets = dataset.targets
        else:
            original_targets = [dataset.imgs[i][1] for i in range(len(dataset))]

        if data_name == 'medmnist':
            num_classes = 9  # medmnist always has 9 classes; forget_class guard not applied
        else:
            num_classes = len(set(original_targets))
            if forget_class is not None and forget_class not in set(original_targets):
                num_classes += 1
            print(num_classes)
    else:
        num_classes = max(train_loader.dataset.targets) + 1
        if forget_class is not None and forget_class not in set(train_loader.dataset.targets):
            num_classes += 1


    if model_name in ('resnet', 'resnet50'):
        cifar_stem = data_name in ('cifar10', 'cifar100')
        model = CustomResNet(num_classes=num_classes, cifar_stem=cifar_stem)
        model = nn.DataParallel(model)
        model.to(device)

    elif model_name == 'vit':
        if data_name in ('cifar100', 'tinyimagenet'):
            model = ViT(num_classes=num_classes)  # native config: vit_base_patch32_224 @ 224x224
        else:
            # Clinical imaging pipeline: reuses a patch16 checkpoint at 512x512,
            # a deliberate override from the checkpoint's native config.
            model = ViT(num_classes=num_classes, timm_model_name='vit_base_patch16_224', img_size=512, patch_size=32)
        model = nn.DataParallel(model)
        model.to(device)

    elif model_name == 'AllCNN':
        if data_name == 'fashionmnist':
            model = AllCNN(n_channels=1, num_classes=num_classes)
        elif data_name in ('medmnist', 'svhn', 'cifar10', 'cifar100'):
            model = AllCNN(n_channels=3, num_classes=num_classes)
        else:
            raise ValueError(f"AllCNN not configured for data_name='{data_name}'")
        model = nn.DataParallel(model)
        model.to(device)

    elif model_name == 'distilbert':
        # Text, not vision - kept in its own module (text_models.py) and
        # imported lazily right here, not at this file's top, so a machine
        # without `transformers` installed never sees an ImportError when
        # training any of the vision model_name branches above.
        from text_models import TextTransformer
        model = TextTransformer(num_classes=num_classes)
        model = nn.DataParallel(model)
        model.to(device)

    else:
        raise ValueError(f"Unknown model_name: '{model_name}'")


    criterion = loss_picker('cross', train_loader=train_loader, device=device, forget_class=forget_class, num_classes=num_classes)
    optimizer = optimizer_picker(optim_name, model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs) if model_name in ('resnet', 'resnet50', 'vit', 'distilbert') else None

    best_acc = 0

    for epo in range(num_epochs):
        print('EPOCH:{}'.format(epo+1))
        loss = train(model=model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, loss_mode='cross',
              device=device)

        print(f'training loss is: {loss}')
        losses.append(loss.item() / len(train_loader))


        _, acc = eval(model=model, data_loader=val_loader, mode='', print_perform=False, device=device)
        accuracies.append(acc.item())
        print('validation acc:{}'.format(acc))

        if acc>=best_acc:
            best_acc = acc

        print('SAVING')
        print(f'current acc = {acc}')
        print(f'best acc = {best_acc}')
        torch.save(model, f'{path}{epo+1}.pth')

        if (epo+1) == num_epochs:
            print('SAVING LAST EPOCH')
            torch.save(model, f'{path}{epo+1}_final_model_{acc}.pth')

        if scheduler is not None:
            scheduler.step()

    end = time.time()
    print('training time:', end-start, 's')

    csv_path = f'{path}training_metrics.csv'
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss', 'Accuracy'])
        for epoch, (loss, acc) in enumerate(zip(losses, accuracies), 1):
            writer.writerow([epoch, loss, acc])

    return model, num_classes, end


def test(model, loader, idx_to_class, num_classes, device):
    """Computes per-class accuracy over a full loader pass, returned as
    {class_name: accuracy}."""
    model.eval()
    correct = [0] * num_classes
    cnt = [0] * num_classes
    class_accuracies = {}


    with torch.no_grad():
        for _, (data, target) in enumerate(tqdm(loader, leave=False)):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            for i in range(target.size(0)):
                label = target[i].item()
                if pred[i].item() == label:
                    correct[label] += 1
                cnt[label] += 1

    for i in range(num_classes):
        accuracy = 0. if cnt[i] == 0 else correct[i] / cnt[i]
        class_name = idx_to_class[i]
        class_accuracies[class_name] = accuracy

    return class_accuracies


def eval(model, data_loader, batch_size=64, mode='backdoor', print_perform=False, device='cpu', name=''):
    """Computes overall accuracy over a full loader pass. Returns
    (sklearn accuracy_score, tensor accuracy) - most callers only use the
    second value. mode/print_perform/name/batch_size are accepted for
    call-site compatibility but not used by this implementation."""
    model.eval()
    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):

        if len(batch_y.shape)>1:
            batch_y=batch_y.squeeze()

        if batch_y.dim() == 0:
            batch_y = batch_y.unsqueeze(0)

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y_predict = model(batch_x)

        batch_y_predict = torch.argmax(batch_y_predict, dim=1)

        y_predict.append(batch_y_predict)
        y_true.append(batch_y)

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)

    num_hits = (y_true == y_predict).float().sum()
    acc = num_hits / y_true.shape[0]


    return accuracy_score(y_true.cpu(), y_predict.cpu()), acc


def train_engine(args, train_remain_loader, val_remain_loader, train_loader, val_loader, \
                 dataset, num_classes, idx_to_class, device, model_name, output_file_name, csv_columns, exp_name='deafault'):
    """Three-way dispatcher called from main.py: --train trains both an
    original and a retrain (remain-only) model from scratch; --retrain_only
    loads an existing original checkpoint and trains just the retrain model;
    otherwise (the usual path for an actual unlearning run) both checkpoints
    are loaded from disk and logged to CSV."""

    if args.train:
        print('=' * 100)
        print(' ' * 25 + 'train original model and retrain model from scratch')
        print('=' * 100)
        ori_model, num_classes, _ = train_save_model(train_loader, val_loader, args.model_name, args.optim_name, args.lr,
                                     args.epoch, device, model_name + "_original_model_", dataset=dataset, data_name=args.data_name, forget_class=None)

        print('\noriginal model acc:\n', test(ori_model, val_loader, idx_to_class, num_classes, device))

        retrain_model, _, _ = train_save_model(train_remain_loader, val_remain_loader, args.model_name, args.optim_name,
                                        args.lr, args.epoch, device, model_name + "_retrain_model_" + 'class_' + str(args.forget_class) + '_', dataset=dataset, data_name=args.data_name, forget_class=args.forget_class)

        print('\nretrain model acc:\n', test(retrain_model, val_remain_loader, idx_to_class, num_classes, device))
        return ori_model, retrain_model, None

    elif args.retrain_only:
        ori_model = load_checkpoint(args.original_model, 'original_model', device=device)
        ori_model.to('cpu')
        print(model_name + "_retrain_" + exp_name + '_' )
        retrain_model, _, time_retrain = train_save_model(train_remain_loader, val_remain_loader, args.model_name, args.optim_name,
                                        args.lr, args.epoch, device,  model_name + "_retrain_" + exp_name + '_' , dataset=dataset, data_name=args.data_name, forget_class=args.forget_class)

        print('\nretrain model acc:\n', test(retrain_model, val_remain_loader, idx_to_class, num_classes, device))

        print(f'RETRAIN TIME {time_retrain}')

        return ori_model, retrain_model, None

    else:
        print('=' * 100)
        print(' ' * 25 + 'load original model and retrain model')
        print('=' * 100)

        # Load and print original model accuracy
        ori_model = load_checkpoint(args.original_model, 'original_model', device=device)

        _, orig_acc = eval(model=ori_model, data_loader=val_loader, mode='', print_perform=False, device=device)

        print('validation acc:{}'.format(orig_acc))
        print('\n ORIGINAL model acc:\n', test(ori_model, val_loader, idx_to_class, num_classes, device))

        ori_model.to('cpu')


        retrain_model = load_checkpoint(args.retrain_model, 'retrain_model', device=device)
        _, retrain_acc = eval(model=retrain_model, data_loader=val_remain_loader, mode='', print_perform=False, device=device)

        print('validation acc retrain:{}'.format(retrain_acc))
        print('\nretrain model acc:\n', test(retrain_model, val_remain_loader, idx_to_class, num_classes, device))

        retrain_model.to('cpu')

        # Log average accuracy of original and retrain models. Forget/remain/
        # test-specific columns are left for the caller to fill in once it
        # runs --run_sota and/or --specific_settings unlearning.
        with open(output_file_name, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            row_data = {
                'Dataset': args.data_name,
                'Model': args.model_name,
                'Forget Acc SOTA': 'N/A',
                'Remain Acc SOTA': 'N/A',
                'Original Acc': orig_acc.detach().item(),
                'Retrain Acc': retrain_acc.detach().item(),
                'Unlearning Time': 'N/A',
                'Per Class Accuracies SOTA': 'N/A'
            }

            writer.writerow(row_data)

        return ori_model, retrain_model, row_data
