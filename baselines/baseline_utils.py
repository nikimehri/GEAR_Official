import os
import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from medmnist import INFO
import medmnist
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

# models.py lives at the repo root, one directory up from baselines/ - add
# it to sys.path so this resolves regardless of the caller's working
# directory (this replaces a hardcoded absolute path to the original
# author's machine).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import *


def seed_torch(seed=2022):
    """Seeds numpy/torch RNGs and forces deterministic cuDNN kernels."""
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(dir_name):
    """Makes dir_name if it doesn't already exist (idempotent)."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def set_num_classes(data_name, dataset):
    """Baselines-subsystem counterpart of utils.set_num_classes, taking a
    plain data_name string instead of an args namespace."""
    if data_name == 'oct_4_class':
        num_classes = 4
    elif data_name == 'fundus_3_class':
        num_classes = 3
    elif data_name == 'open_source':
        num_classes = 3
    elif data_name == 'oculoplastic':
        num_classes = 3
    elif data_name == 'dr_grade':
        num_classes = 5
    elif data_name == 'mri':
        num_classes = 4
    elif data_name == 'ultrasound':
        num_classes = 3
    elif data_name == 'cxr':
        num_classes = 4
    elif data_name == 'medmnist':
        num_classes = 9
    elif data_name == 'cifar100':
        num_classes = 100
    elif data_name == 'tinyimagenet':
        num_classes = 200
    else:
        num_classes = 10

    if data_name == 'medmnist':
        info = INFO['pathmnist']
        num_classes = len(info['label'])
        print(info['label'])
        idx_to_class = {i: info['label'][str(i)] for i in range(num_classes)}
    else:
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    return num_classes, idx_to_class




def split_class_data(dataset, forget_class, num_forget):
    """Splits a dataset's indices into forget/remain/class_remain (leftover
    same-class samples beyond num_forget) index lists."""
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




def get_unlearn_loader(trainset, testset, forget_class, batch_size, num_forget, selective_unlearning = False, repair_num_ratio=0.01):
    """Baselines-subsystem counterpart of make_dataloaders.get_unlearn_loader."""
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




def get_custom_forget_loader_oculoplastics(dataset, metadata_dict, batch_size=8):
    """Baselines-subsystem counterpart of
    make_dataloaders.get_custom_forget_loader_oculoplastics."""
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




def get_custom_forget_loader(dataset, metadata_dict, attribute_to_forget, batch_size=8):
    """Baselines-subsystem counterpart of
    make_dataloaders.get_custom_forget_loader."""
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


def dataloader_engine(batch_size, trainset, testset, combined_df, num_forget=5000, forget_class = 0, custom_unlearn=False, oculoplastics = False, selective_unlearning = False):
    """Baselines-subsystem entry point used by baseline_main.py - builds
    train/test forget+remain loaders (no validation split, unlike
    make_dataloaders.dataloader_engine; note the positional argument order
    also differs between the two)."""
    if custom_unlearn and not oculoplastics:
        print('Getting CUSTOM Unlearn Loader using OHE of Metadata')
        train_dict = map_metadata(trainset, combined_df)
        test_dict = map_metadata(testset, combined_df)
        
        train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
        train_forget_index, train_remain_index, test_forget_index, test_remain_index \
            = get_custom_unlearn_loader(trainset, testset, train_dict, test_dict, 'Cirrus 800 FA', batch_size)
        
    elif custom_unlearn and oculoplastics:
        print('Getting CUSTOM Unlearn Loader using OHE of Metadata')
        train_dict = map_metadata_oculoplastics(trainset, combined_df)
        test_dict = map_metadata_oculoplastics(testset, combined_df)
        
        train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
        train_forget_index, train_remain_index, test_forget_index, test_remain_index \
            = get_custom_unlearn_loader_oculoplastics(trainset, testset, train_dict, test_dict, batch_size)

    
    else:
        print('getting unlearn loader')
        num_forget = num_forget
        train_dict = None
        test_dict = None
        train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
        train_forget_index, train_remain_index, test_forget_index, test_remain_index \
            = get_unlearn_loader(trainset, testset, forget_class, batch_size, num_forget, selective_unlearning=selective_unlearning)
        
    return train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
        train_forget_index, train_remain_index, test_forget_index, test_remain_index, train_dict, test_dict

def map_metadata(dataset, df):
    """Baselines-subsystem counterpart of utils.map_metadata."""
    metadata_dict = {}
    print('in map metadata')
    print(len(dataset.dataset.imgs))
    for img_path, _ in dataset.dataset.imgs:  
        filename = os.path.basename(img_path)
        row = df[df['de_FileName'] == filename]
        if not row.empty:
            ohe_vector = create_ohe_vector(row)
            metadata_dict[filename] = ohe_vector
    
    return metadata_dict

def get_custom_unlearn_loader_oculoplastics(trainset, testset, train_dict, test_dict, batch_size, num_forget=1000, repair_num_ratio=0.01):
    """Baselines-subsystem counterpart of
    make_dataloaders.get_custom_unlearn_loader_oculoplastics."""
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





def split_metadata_data_oculoplastics(subset, metadata_dict, num_forget):
    """Baselines-subsystem counterpart of
    make_dataloaders.split_metadata_data_oculoplastics."""
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


def create_ohe_vector(row):
    """Baselines-subsystem counterpart of utils.create_ohe_vector."""
    attributes = ['OS', 'OD', 'Spectralis (Scans)', 'Cirrus 800 FA', '2015', '2016', '2017', '2018']

    ohe_vector = [0] * len(attributes)

    if row['WhichEye'].values[0] == 'OS':
        ohe_vector[0] = 1
    elif row['WhichEye'].values[0] == 'OD':
        ohe_vector[1] = 1

    if 'Spectralis' in row['DeviceProc'].values[0]:
        ohe_vector[2] = 1
    elif 'Cirrus 800 FA' in row['DeviceProc'].values[0]:
        ohe_vector[3] = 1

    date_str = row['ExamDate'].values[0]
    try:
        exam_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        year = str(exam_date.year)
        if year in attributes[4:]:
            index = attributes.index(year)
            ohe_vector[index] = 1
    except ValueError as e:
        print(f"Error parsing date: {date_str} - {e}")

    return ohe_vector





def map_metadata_oculoplastics(dataset, df, feature='vert_pf', threshold=11):
    """Baselines-subsystem counterpart of
    make_dataloaders.map_metadata_oculoplastics."""
    metadata_dict = {}
    for img_path, _ in dataset.dataset.imgs:
        filename = os.path.basename(img_path)
        row = df[df['file'] == filename[:-9]]
        # print(row)
        if not row.empty:
            left_feature = row[f'left_{feature}'].values[0]
            right_feature = row[f'right_{feature}'].values[0]
            avg_feature = (left_feature + right_feature) / 2
            if avg_feature > threshold:
                metadata_dict[filename] = avg_feature
    print(f"Total images mapped with {feature} > {threshold}: {len(metadata_dict)}")
    return metadata_dict





def get_custom_unlearn_loader(trainset, testset, train_dict, test_dict, unlearn_attribute, batch_size):
    """Baselines-subsystem counterpart of
    make_dataloaders.get_custom_unlearn_loader."""
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


def split_metadata_data(subset, metadata_dict, unlearn_attribute, num_forget):
    """Baselines-subsystem counterpart of make_dataloaders.split_metadata_data."""
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



def resize_width_pad_height(target_width=512, target_height=512):
    """Baselines-subsystem counterpart of
    make_dataloaders.resize_width_pad_height."""
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



def get_dataset(data_name, path='./data', model_name=None):
    """Baselines-subsystem counterpart of make_dataloaders.get_dataset - a
    more limited dataset zoo (no mnist/svhn/oculoplastic-specific branch;
    the else branch covers any other ImageFolder-compatible directory).

    model_name is only consulted for cifar100/tinyimagenet: 'vit' resizes to
    224x224 with ImageNet stats instead of each dataset's native pipeline -
    see make_dataloaders.get_dataset's docstring for why."""

    if data_name == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 normalization
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
        if model_name == 'vit':
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        trainset = datasets.CIFAR100(root=path, train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR100(root=path, train=False, download=True, transform=test_transform)
        dataset = datasets.CIFAR100(root=path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        return trainset, testset, dataset

    elif data_name == 'tinyimagenet':
        root = os.path.join(path, 'tiny-imagenet-200')
        train_dir = os.path.join(root, 'train')
        val_dir = os.path.join(root, 'val')
        if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
            raise FileNotFoundError(
                f"TinyImageNet not found at {root}. Run "
                f"`python scripts/prepare_tinyimagenet.py --dataset_dir {path}` first."
            )
        if os.path.isdir(os.path.join(val_dir, 'images')):
            # val/ still has its original flat layout (val/images/*.JPEG) -
            # prepare_tinyimagenet.py's reorganize_val() hasn't run against
            # this path yet. Loading it as-is via ImageFolder would silently
            # produce a single-class dataset (all 10000 images under one
            # 'images' folder) instead of failing here.
            raise FileNotFoundError(
                f"TinyImageNet val/ at {val_dir} hasn't been reorganized into "
                f"per-class subdirectories yet. Run "
                f"`python scripts/prepare_tinyimagenet.py --dataset_dir {path}` "
                f"to fix it (safe to re-run)."
            )

        if model_name == 'vit':
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
            ])

        trainset = datasets.ImageFolder(train_dir, transform=train_transform)
        testset = datasets.ImageFolder(val_dir, transform=test_transform)
        dataset = datasets.ImageFolder(val_dir, transform=transforms.Compose([transforms.ToTensor()]))
        return trainset, testset, dataset

    elif data_name == 'fashionmnist':
        train_transform = transforms.Compose([
            transforms.Resize(32), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(32),  # Resize to 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Grayscale normalization
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
            transforms.Resize(32),  # Resize to 32x32
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


def get_forget_loader(dt, forget_class):
    idx = []
    els_idx = []
    for i in range(len(dt)):
        _, lbl = dt[i]
        if lbl == forget_class:
            idx.append(i)
        else:
            els_idx.append(i)
    forget_loader = torch.utils.data.DataLoader(dt, batch_size=8, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(idx), drop_last=True)
    remain_loader = torch.utils.data.DataLoader(dt, batch_size=8, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(els_idx), drop_last=True)
    return forget_loader, remain_loader





def load_model(model_type, num_classes, data_name, n_channels=3, size=32, batch_norm=True):
    """Constructs an AllCNN or CustomResNet by string name. cifar_stem is
    applied to both resnet and resnet50 (matching trainer.py's model
    construction) so a CIFAR-trained ResNet always gets the small-image
    stem regardless of which of the two names it was saved under."""
    if model_type == 'allcnn':
        if data_name == 'fashionmnist':
            n_channels = 1
        model = AllCNN(n_channels=n_channels, num_classes=num_classes, size=size, batch_norm=batch_norm)
    elif model_type in ('resnet', 'resnet50'):
        cifar_stem = data_name in ('cifar10', 'cifar100')
        model = CustomResNet(num_classes=num_classes, cifar_stem=cifar_stem)
    elif model_type == 'vit':
        if data_name in ('cifar100', 'tinyimagenet'):
            model = ViT(num_classes=num_classes)
        else:
            model = ViT(num_classes=num_classes, timm_model_name='vit_base_patch16_224', img_size=512, patch_size=32)
    return model


def load_model_state(model, checkpoint_path, map_location=None):
    """Loads a checkpoint (full model object, DataParallel-wrapped model, or
    raw state dict) and applies its weights to model, stripping any
    'module.' DataParallel prefix. weights_only=False is required since
    these checkpoints are full pickled model objects, not plain state dicts."""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=map_location)

    if isinstance(checkpoint, torch.nn.DataParallel):
        state_dict = checkpoint.module.state_dict()
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()

    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    return model




def load_checkpoint_without_dataparallel(checkpoint_path, model):
    """Used for the pre-baked Chen/Ravi competitor checkpoints: same
    DataParallel/state-dict normalization as load_model_state, via a
    slightly different code path."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if isinstance(checkpoint, torch.nn.DataParallel):
        checkpoint = checkpoint.module.state_dict()
    elif not isinstance(checkpoint, dict):  
        checkpoint = checkpoint.state_dict()
    if "module." in list(checkpoint.keys())[0]:  
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            new_key = k.replace("module.", "") 
            new_state_dict[new_key] = v
        checkpoint = new_state_dict
    model.load_state_dict(checkpoint)
    return model




def cm_score(estimator, X, y):
    """Custom sklearn scorer (accuracy) used as evaluate_attack_model's
    cross_val_score scoring function."""
    y_pred = estimator.predict(X)
    return accuracy_score(y, y_pred)


def evaluate_attack_model(sample_loss,
                          members,
                          n_splits = 5,
                          random_state = None):
  """Computes the cross-validation score of a membership inference attack.
  Args:
    sample_loss : array_like of shape (n,).
      objective function evaluated on n samples.
    members : array_like of shape (n,),
      whether a sample was used for training.
    n_splits: int
      number of splits to use in the cross-validation.
    random_state: int, RandomState instance or None, default=None
      random state to use in cross-validation splitting.
  Returns:
    score : array_like of size (n_splits,)
  """

  unique_members = np.unique(members)
  if not np.all(unique_members == np.array([0, 1])):
    raise ValueError("members should only have 0 and 1s")

  attack_model = LogisticRegression()
  cv = StratifiedShuffleSplit(
      n_splits=n_splits, random_state=random_state)
  return cross_val_score(attack_model, sample_loss, members, cv=cv, scoring=cm_score)



def membership_inference_attack(model, test_loader, forget_loader, device, seed=2022, name='mia'):
    """
    Simple MIA function that:
      - Computes losses on the entire test set (test_loader).
      - Computes losses on the forget class subset (forget_loader).
      - Then performs a membership inference attack using logistic regression.
    """

    model.eval()
    cr = nn.CrossEntropyLoss(reduction="none")

    # 1) Compute test_losses
    test_losses = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        if target.dim() > 1:
            target = target.squeeze()
        
        output = model(data)
        loss = cr(output, target)
        test_losses.extend(loss.detach().cpu().numpy())

    # 2) Compute forget_losses
    forget_losses = []
    for data, target in forget_loader:
        data, target = data.to(device), target.to(device)
        
        if target.dim() > 1:
            target = target.squeeze()
        
        output = model(data)
        loss = cr(output, target)
        forget_losses.extend(loss.detach().cpu().numpy())

    np.random.seed(seed)
    random.seed(seed)

    if len(forget_losses) > len(test_losses):
        forget_losses = random.sample(forget_losses, len(test_losses))
    elif len(test_losses) > len(forget_losses):
        test_losses = random.sample(test_losses, len(forget_losses))

    # 4) Plot histograms
    sns.histplot(test_losses, kde=False, label="test-loss", color="blue")
    sns.histplot(forget_losses, kde=False, label="forget-loss", color="orange")
    plt.legend(prop={"size": 14})
    plt.title("Loss Histograms", size=18)
    plt.xlabel("Loss Values", size=14)
    plt.savefig(f"mia_loss_histogram_{name}_seed{seed}.jpg", dpi=300)
    plt.close()

    # 5) Evaluate MIA using logistic regression
    test_labels = [0] * len(test_losses)     # 0 → Non-member
    forget_labels = [1] * len(forget_losses) # 1 → Member
    features = np.array(test_losses + forget_losses).reshape(-1, 1)
    labels = np.array(test_labels + forget_labels)
    features = np.clip(features, -100, 100)

    score = evaluate_attack_model(features, labels, n_splits=5, random_state=seed)
    return score

def eval(model, data_loader, batch_size=64, device='cpu', name=''):
    """Overall accuracy over a full loader pass. Returns
    (sklearn accuracy_score, tensor accuracy)."""
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

        # batch_y = torch.argmax(batch_y, dim=1)
        y_predict.append(batch_y_predict)
        y_true.append(batch_y)

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)

    num_hits = (y_true == y_predict).float().sum()
    acc = num_hits / y_true.shape[0]
    
    return accuracy_score(y_true.cpu(), y_predict.cpu()), acc
