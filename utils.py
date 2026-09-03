import os
import random
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from datetime import datetime
import csv
from medmnist import INFO

def seed_torch(seed=2022):
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def set_up_save(args, distributions, gamma_values, l1_norms, name):

    output_file_name = name + '.csv'
    
    csv_columns = ['Dataset', 'Model', 'Gamma', 'Distribution', 'Forget Acc SOTA', 'Remain Acc SOTA', 'Original Acc', 'Retrain Acc', 'Unlearning Time', 'Per Class Accuracies SOTA']
    # csv_columns = ['Dataset',  'Original Acc', 'Retrain Acc', 'Unlearning Time']

    for dist in distributions:
        for gamma in gamma_values:
            csv_columns.append(f'Forget Acc {dist} {gamma}')
            csv_columns.append(f'Remain Acc {dist} {gamma}')
            csv_columns.append(f'Per Class Accuracies {dist} {gamma}')
        if len(l1_norms)>1:
            for setting in l1_norms:
                for gamma in gamma_values:
                    csv_columns.append(f'Forget Acc {dist}_lambda_{gamma}_{setting}')
                    csv_columns.append(f'Remain Acc {dist}_lambda_{gamma}_{setting}')
                    csv_columns.append(f'Per Class Accuracies {dist}_lambda_{gamma}_{setting}')

    with open(output_file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
    
    return csv_columns, output_file_name


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def map_metadata(dataset, df):
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

def create_ohe_vector(row):
    attributes = ['OS', 'OD', 'Spectralis (Scans)', 'Cirrus 800 FA', '2015', '2016', '2017', '2018']

    ohe_vector = [0] * len(attributes)
    
    # Eye side (OS/OD)
    if row['WhichEye'].values[0] == 'OS':
        ohe_vector[0] = 1
    elif row['WhichEye'].values[0] == 'OD':
        ohe_vector[1] = 1
    
    # Device
    if 'Spectralis' in row['DeviceProc'].values[0]:
        ohe_vector[2] = 1
    elif 'Cirrus 800 FA' in row['DeviceProc'].values[0]:
        ohe_vector[3] = 1

    # Exam Date
    date_str = row['ExamDate'].values[0]
    try:
        # Adjust the format string to match the actual format of your dates
        exam_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        year = str(exam_date.year)
        if year in attributes[4:]:
            index = attributes.index(year)
            ohe_vector[index] = 1
    except ValueError as e:
        print(f"Error parsing date: {date_str} - {e}")

    return ohe_vector

def set_num_classes(args, dataset):
    if args.data_name == 'oct_4_class':
        num_classes = 4
    elif args.data_name == 'fundus_3_class':
        num_classes = 3
    elif args.data_name == 'open_source':
        num_classes = 3
    elif args.data_name == 'oculoplastic':
        num_classes = 3
    elif args.data_name == 'dr_grade':
        num_classes = 5
    elif args.data_name == 'mri':
        num_classes = 4
    elif args.data_name == 'ultrasound':
        num_classes = 3
    elif args.data_name == 'cxr':
        num_classes = 4
    elif args.data_name == 'medmnist':
        num_classes = 9
    elif args.data_name == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10

    if args.data_name == 'svhn':
        num_classes = 10
        idx_to_class = {i: str(i) for i in range(10)}
    elif args.data_name == 'medmnist':
        info = INFO['pathmnist']
        num_classes = len(info['label'])
        print(info['label'])
        idx_to_class = {i: info['label'][str(i)] for i in range(num_classes)}
    else:
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    return num_classes, idx_to_class