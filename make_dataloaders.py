import os
import random
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from datetime import datetime
from utils import *
from medmnist import INFO, Evaluator
import medmnist
from torchvision import datasets, transforms

def resize_width_pad_height(target_width=512, target_height=512):
    def transform(image):
        aspect_ratio = image.width / image.height
        new_height = int(round(target_width / aspect_ratio))

        resize_transform = transforms.Resize((new_height, target_width))
        resized_image = resize_transform(image)

        padding_top = (target_height - new_height) // 2
        padding_bottom = target_height - new_height - padding_top

        pad_transform = transforms.Pad((0, padding_top, 0, padding_bottom), fill=0, padding_mode='constant')
        padded_image = pad_transform(resized_image)

        return padded_image

    return transform

def get_dataset(data_name, path='./data'):
    if data_name == 'dr_grade':
        img_size = 36
    else:
        img_size = 512

    if data_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST(root=path, train=True, download=True, transform=transform)
        testset = datasets.MNIST(root=path, train=False, download=True, transform=transform)
        dataset = datasets.MNIST(root=path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        return trainset, testset, dataset

    elif data_name == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        trainset = datasets.CIFAR10(root=path, train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR10(root=path, train=False, download=True, transform=test_transform)
        dataset = datasets.CIFAR10(root=path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        return trainset, testset, dataset

    elif data_name == 'cifar100':
        # CIFAR-100 has 100 classes over the same 32x32 image size as CIFAR-10,
        # so it needs its own normalization stats and slightly heavier
        # augmentation (rotation + color jitter) to compensate for the smaller
        # per-class sample count (500 images/class vs. CIFAR-10's 5000).
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])
        trainset = datasets.CIFAR100(root=path, train=True,  download=True, transform=train_transform)
        testset  = datasets.CIFAR100(root=path, train=False, download=True, transform=test_transform)
        dataset  = datasets.CIFAR100(root=path, train=False, download=True,
                                     transform=transforms.Compose([transforms.ToTensor()]))
        return trainset, testset, dataset

    elif data_name == 'svhn':
        # SVHN has 3 channels and 32x32 images
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])

        trainset = datasets.SVHN(root=path, split='train', download=True, transform=train_transform)
        testset = datasets.SVHN(root=path, split='test', download=True, transform=test_transform)
        dataset = datasets.SVHN(root=path, split='test', download=True, transform=test_transform)
        return trainset, testset, dataset

    elif data_name == 'fashionmnist':
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        trainset = datasets.FashionMNIST(root=path, train=True, download=True, transform=train_transform)
        testset = datasets.FashionMNIST(root=path, train=False, download=True, transform=test_transform)
        dataset = datasets.FashionMNIST(root=path, train=False, download=True, transform=test_transform)
        return trainset, testset, dataset

    elif data_name == 'medmnist':
        info = INFO['pathmnist']

        n_channels = info['n_channels']
        n_classes = len(info['label'])
        print(n_classes, n_channels)
        DataClass = getattr(medmnist, info['python_class'])

        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        trainset = DataClass(split='train', transform=train_transform, download=True)
        testset = DataClass(split='test', transform=test_transform, download=True)
        dataset = DataClass(split='train', transform=transforms.Compose([transforms.ToTensor()]))
        return trainset, testset, dataset


    elif data_name == 'oculoplastic':
            print('IN OCULOPLASTIC')
            print(path)
            train_transforms = transforms.Compose([
                resize_width_pad_height(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomResizedCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            test_transforms = transforms.Compose([
                resize_width_pad_height(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            dataset = datasets.ImageFolder(path)
            num_train = len(dataset)
            indices = list(range(num_train))

            split = int(np.floor(0.2 * num_train))
            np.random.shuffle(indices)

            train_idx, test_idx = indices[split:], indices[:split]

            train_dataset = datasets.ImageFolder(path, transform=train_transforms)
            test_dataset = datasets.ImageFolder(path, transform=test_transforms)

            trainset = torch.utils.data.Subset(train_dataset, train_idx)
            testset = torch.utils.data.Subset(test_dataset, test_idx)

            return trainset, testset, dataset

    else:
        train_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(path)
        num_train = len(dataset)
        indices = list(range(num_train))

        split = int(np.floor(0.2 * num_train))
        np.random.shuffle(indices)

        train_idx, test_idx = indices[split:], indices[:split]

        train_dataset = datasets.ImageFolder(path, transform=train_transforms)
        test_dataset = datasets.ImageFolder(path, transform=test_transforms)

        trainset = torch.utils.data.Subset(train_dataset, train_idx)
        testset = torch.utils.data.Subset(test_dataset, test_idx)

        return trainset, testset, dataset

def get_dataloader(trainset, testset, batch_size, device):

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def split_class_data(dataset, forget_class, num_forget):
    forget_index = []
    class_remain_index = []
    remain_index = []
    sum = 0

    for i, (data, target) in enumerate(dataset):
        if target == forget_class and sum < num_forget:
            forget_index.append(i)

            sum += 1
        elif target == forget_class and sum >= num_forget:
            class_remain_index.append(i)
            remain_index.append(i)
            sum += 1

        else:
            remain_index.append(i)


    return forget_index, remain_index, class_remain_index


def split_metadata_data(subset, metadata_dict, unlearn_attribute, num_forget):
    attributes = ['OS', 'OD', 'Spectralis (Scans)', 'Cirrus 800 FA', '2015', '2016', '2017', '2018']
    attr_index = attributes.index(unlearn_attribute)

    forget_index = []
    class_remain_index = []
    remain_index = []
    sum = 0

    original_dataset = subset.dataset
    for i, subset_index in enumerate(subset.indices):
        img_path, _ = original_dataset.imgs[subset_index]
        filename = os.path.basename(img_path)
        if filename in metadata_dict:
            ohe_vector = metadata_dict[filename]
            if ohe_vector[attr_index] == 1 and sum < num_forget:
                forget_index.append(i)
                sum += 1
            elif ohe_vector[attr_index] == 1 and sum >= num_forget:
                class_remain_index.append(i)
                remain_index.append(i)
                sum += 1
            else:
                remain_index.append(i)

    return forget_index, remain_index, class_remain_index


def get_custom_unlearn_loader(trainset, testset, train_dict, test_dict, unlearn_attribute, batch_size):
    num_forget = 1000
    repair_num_ratio = 0.01

    train_forget_index, train_remain_index, class_remain_index = split_metadata_data(
        trainset, train_dict, unlearn_attribute, num_forget)

    test_forget_index, test_remain_index, _ = split_metadata_data(
        testset, test_dict, unlearn_attribute, num_forget=len(testset.dataset.imgs))

    repair_class_index = random.sample(class_remain_index, int(repair_num_ratio * len(class_remain_index)))

    train_forget_sampler = SubsetRandomSampler(train_forget_index)
    train_remain_sampler = SubsetRandomSampler(train_remain_index)

    repair_class_sampler = SubsetRandomSampler(repair_class_index)

    test_forget_sampler = SubsetRandomSampler(test_forget_index)
    test_remain_sampler = SubsetRandomSampler(test_remain_index)

    train_forget_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, sampler=train_forget_sampler)
    train_remain_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, sampler=train_remain_sampler)

    repair_class_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, sampler=repair_class_sampler)

    test_forget_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, sampler=test_forget_sampler)
    test_remain_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, sampler=test_remain_sampler)

    return train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
           train_forget_index, train_remain_index, test_forget_index, test_remain_index


def get_unlearn_loader(trainset, testset, forget_class, batch_size, num_forget, repair_num_ratio=0.01, selective_unlearning = False):

    train_forget_index, train_remain_index, class_remain_index = split_class_data(trainset, forget_class,
                                                                                  num_forget=num_forget)

    if not selective_unlearning:
        test_forget_index, test_remain_index, _ = split_class_data(testset, forget_class, num_forget=len(testset))
    else:
        test_forget_index = train_forget_index
        test_remain_index = train_remain_index


    repair_class_index = random.sample(class_remain_index, int(repair_num_ratio * len(class_remain_index)))

    train_forget_sampler = SubsetRandomSampler(train_forget_index)  # 5000
    train_remain_sampler = SubsetRandomSampler(train_remain_index)  # 45000

    repair_class_sampler = SubsetRandomSampler(repair_class_index)

    test_forget_sampler = SubsetRandomSampler(test_forget_index)  # 1000
    test_remain_sampler = SubsetRandomSampler(test_remain_index)  # 9000

    train_forget_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=train_forget_sampler)
    train_remain_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=train_remain_sampler)

    repair_class_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=repair_class_sampler)

    test_forget_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                     sampler=test_forget_sampler)
    test_remain_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                     sampler=test_remain_sampler)

    return train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
           train_forget_index, train_remain_index, test_forget_index, test_remain_index


def get_relearn_loader_from_remain(train_remain_loader, batch_size, metadata_df, exclude_feature='Cirrus 800 FA'):
    class_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    relearn_indices = []

    original_dataset = train_remain_loader.dataset.dataset

    for idx in train_remain_loader.dataset.indices:
        img_path, target = original_dataset.imgs[idx]
        filename = os.path.basename(img_path)
        if target == 0:  # DR class
            relearn_indices.append(idx)
        elif target == 1 and filename in metadata_df['de_FileName'].values:
            if metadata_df[metadata_df['de_FileName'] == filename]['DeviceProc'].values[0] != exclude_feature:
                relearn_indices.append(idx)

    relearn_dataset = torch.utils.data.Subset(original_dataset, relearn_indices)
    relearn_dataset.dataset.transform = class_transforms

    relearn_loader = DataLoader(relearn_dataset, batch_size=batch_size, shuffle=True)

    return relearn_loader


def get_forget_loader(dt, forget_class):
    idx = []
    els_idx = []
    count = 0
    for i in range(len(dt)):
        _, lbl = dt[i]
        if lbl == forget_class:
            # if forget:
            #     count += 1
            #     if count > forget_num:
            #         continue
            idx.append(i)
        else:
            els_idx.append(i)
    forget_loader = torch.utils.data.DataLoader(dt, batch_size=8, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(idx), drop_last=True)
    remain_loader = torch.utils.data.DataLoader(dt, batch_size=8, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(els_idx), drop_last=True)
    return forget_loader, remain_loader


def get_custom_forget_loader(dataset, metadata_dict, attribute_to_forget, batch_size=8):
    forget_indices = []
    remain_indices = []

    attribute_index = {
        'OS': 0, 'OD': 1, 'Spectralis (Scans)': 2, 'Cirrus 800 FA': 3,
        '2015': 4, '2016': 5, '2017': 6, '2018': 7
    }.get(attribute_to_forget)

    original_dataset = dataset.dataset


    if attribute_index is None:
        raise ValueError(f"Attribute {attribute_to_forget} not recognized.")

    for i, subset_index in enumerate(dataset.indices):
        img_path, _ = original_dataset.imgs[subset_index]
        filename = os.path.basename(img_path)

        if filename in metadata_dict:
            ohe_vector = metadata_dict[filename]
            if ohe_vector[attribute_index] == 1:
                forget_indices.append(i)
            else:
                remain_indices.append(i)


    forget_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(forget_indices), drop_last=True)
    remain_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(remain_indices), drop_last=True)

    return forget_loader, remain_loader


def get_custom_forget_loader_oculoplastics(dataset, metadata_dict, batch_size=8):
    forget_indices = []
    remain_indices = []

    original_dataset = dataset.dataset

    for i, subset_index in enumerate(dataset.indices):
        img_path, _ = original_dataset.imgs[subset_index]
        filename = os.path.basename(img_path)

        if filename in metadata_dict:
            forget_indices.append(i)
        else:
            remain_indices.append(i)

    print(f'length of TEST forget indices :{len(forget_indices)}')
    print(f'length of TEST remain indices :{len(remain_indices)}')

    forget_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(forget_indices), drop_last=True)
    remain_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(remain_indices), drop_last=True)

    print(f'length of TEST forget loader :{len(forget_loader)}')
    print(f'length of TEST remain loader :{len(remain_loader)}')

    return forget_loader, remain_loader


def dataloader_engine(args, trainset, valset, testset, combined_df=None, num_forget=5000,
                      oculoplastics=False, selective_unlearning=False):
    """Builds forget/remain loaders for the train, validation, and test splits.
    Each split is fed through the same lower-level loader-building function
    twice (as both the 'trainset' and 'testset' argument) purely to reuse that
    function's forget/remain-splitting logic on a single dataset — the
    duplicate half of each call's return value is discarded."""

    if args.custom_unlearn and not oculoplastics:
        print('Getting CUSTOM Unlearn Loader using OHE of Metadata')

        train_dict = map_metadata(trainset, combined_df)
        val_dict = map_metadata(valset, combined_df)
        test_dict = map_metadata(testset, combined_df)

        train_forget_loader, train_remain_loader, _, _, _, \
        train_forget_index, train_remain_index, _, _ = get_custom_unlearn_loader(
            trainset, trainset, train_dict, train_dict, args.to_forget, args.batch_size
        )

        val_forget_loader, val_remain_loader, _, _, _, \
        val_forget_index, val_remain_index, _, _ = get_custom_unlearn_loader(
            valset, valset, val_dict, val_dict, args.to_forget, args.batch_size
        )

        test_forget_loader, test_remain_loader, _, _, _, \
        test_forget_index, test_remain_index, _, _ = get_custom_unlearn_loader(
            testset, testset, test_dict, test_dict, args.to_forget, args.batch_size
        )

    elif args.custom_unlearn and oculoplastics:
        print('Getting CUSTOM Unlearn Loader using OHE of Metadata')

        train_dict = map_metadata_oculoplastics(trainset, combined_df)
        val_dict = map_metadata_oculoplastics(valset, combined_df)
        test_dict = map_metadata_oculoplastics(testset, combined_df)

        train_forget_loader, train_remain_loader, _, _, _, \
        train_forget_index, train_remain_index, _, _ = get_custom_unlearn_loader_oculoplastics(
            trainset, trainset, train_dict, train_dict, args.batch_size
        )

        val_forget_loader, val_remain_loader, _, _, _, \
        val_forget_index, val_remain_index, _, _ = get_custom_unlearn_loader_oculoplastics(
            valset, valset, val_dict, val_dict, args.batch_size
        )

        test_forget_loader, test_remain_loader, _, _, _, \
        test_forget_index, test_remain_index, _, _ = get_custom_unlearn_loader_oculoplastics(
            testset, testset, test_dict, test_dict, args.batch_size
        )

    else:
        print('getting unlearn loader')
        forget_class = args.forget_class
        print(forget_class, num_forget)

        train_dict = None
        val_dict = None
        test_dict = None

        train_forget_loader, train_remain_loader, _, _, _, \
        train_forget_index, train_remain_index, _, _ = get_unlearn_loader(
            trainset, trainset, forget_class, args.batch_size, num_forget,
            selective_unlearning=selective_unlearning
        )

        val_forget_loader, val_remain_loader, _, _, _, \
        val_forget_index, val_remain_index, _, _ = get_unlearn_loader(
            valset, valset, forget_class, args.batch_size, num_forget,
            selective_unlearning=selective_unlearning
        )

        test_forget_loader, test_remain_loader, _, _, _, \
        test_forget_index, test_remain_index, _, _ = get_unlearn_loader(
            testset, testset, forget_class, args.batch_size, num_forget,
            selective_unlearning=selective_unlearning
        )

    return (
        train_forget_loader, train_remain_loader,
        val_forget_loader, val_remain_loader,
        test_forget_loader, test_remain_loader,
        train_forget_index, train_remain_index,
        val_forget_index, val_remain_index,
        test_forget_index, test_remain_index,
        train_dict, val_dict, test_dict
    )


def map_metadata_oculoplastics(dataset, df, feature='vert_pf', threshold=11):
    metadata_dict = {}
    for img_path, _ in dataset.dataset.imgs:
        filename = os.path.basename(img_path)
        row = df[df['file'] == filename[:-9]]
        if not row.empty:
            left_feature = row[f'left_{feature}'].values[0]
            right_feature = row[f'right_{feature}'].values[0]
            avg_feature = (left_feature + right_feature) / 2
            if avg_feature > threshold:
                metadata_dict[filename] = avg_feature
    print(f"Total images mapped with {feature} > {threshold}: {len(metadata_dict)}")
    return metadata_dict



def split_metadata_data_oculoplastics(subset, metadata_dict, num_forget):
    forget_index = []
    remain_index = []
    sum = 0

    original_dataset = subset.dataset
    for i, subset_index in enumerate(subset.indices):
        img_path, _ = original_dataset.imgs[subset_index]
        filename = os.path.basename(img_path)
        if filename in metadata_dict:
            if sum < num_forget:
                forget_index.append(i)
                sum += 1
            else:
                remain_index.append(i)
        else:
            remain_index.append(i)

    print(f"Total images to forget: {len(forget_index)}")
    print(f"Total images to remain: {len(remain_index)}")
    return forget_index, remain_index

def get_custom_unlearn_loader_oculoplastics(trainset, testset, train_dict, test_dict, batch_size, num_forget=1000, repair_num_ratio=0.01):
    train_forget_index, train_remain_index = split_metadata_data_oculoplastics(trainset, train_dict, num_forget)
    test_forget_index, test_remain_index = split_metadata_data_oculoplastics(testset, test_dict, num_forget=len(testset.dataset.imgs))

    repair_class_index = random.sample(train_remain_index, int(repair_num_ratio * len(train_remain_index)))

    train_forget_sampler = SubsetRandomSampler(train_forget_index)
    train_remain_sampler = SubsetRandomSampler(train_remain_index)
    repair_class_sampler = SubsetRandomSampler(repair_class_index)
    test_forget_sampler = SubsetRandomSampler(test_forget_index)
    test_remain_sampler = SubsetRandomSampler(test_remain_index)

    train_forget_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, sampler=train_forget_sampler)
    train_remain_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, sampler=train_remain_sampler)
    repair_class_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, sampler=repair_class_sampler)
    test_forget_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, sampler=test_forget_sampler)
    test_remain_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, sampler=test_remain_sampler)

    print(f"Train forget loader size: {len(train_forget_loader)}")
    print(f"Train remain loader size: {len(train_remain_loader)}")
    print(f"Repair class loader size: {len(repair_class_loader)}")
    print(f"Test forget loader size: {len(test_forget_loader)}")
    print(f"Test remain loader size: {len(test_remain_loader)}")

    return train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
           train_forget_index, train_remain_index, test_forget_index, test_remain_index
