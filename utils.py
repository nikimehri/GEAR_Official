import os
import torch
import numpy as np
from datetime import datetime
import csv
from medmnist import INFO

def seed_torch(seed=2022):
    """Seeds numpy/torch RNGs and forces deterministic cuDNN kernels, for
    run-to-run reproducibility."""
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def set_up_save(args, name):
    """Creates {name}.csv and writes its header. Columns cover the two
    stages a run can report: the SOTA/baseline unlearning run (--run_sota)
    and the GEAR-specific run (--specific_settings). Note this truncates
    any existing file with the same name."""

    output_file_name = name + '.csv'

    csv_columns = [
        'Dataset', 'Model', 'Original Acc', 'Retrain Acc',
        'Forget Acc SOTA', 'Remain Acc SOTA', 'Per Class Accuracies SOTA', 'Unlearning Time',
        'Forget Acc', 'Retain Remote Acc', 'Retain Adjacent Acc', 'Test Acc', 'MIA', 'AIN',
    ]

    with open(output_file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

    return csv_columns, output_file_name


def create_dir(dir_name):
    """Makes dir_name if it doesn't already exist (idempotent)."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def map_metadata(dataset, df):
    """Builds {image filename: one-hot metadata vector} for every image in
    an ImageFolder-based Subset, by matching against a clinical metadata
    dataframe's filename column. Used for the --custom_unlearn (attribute-
    based, not class-based) forgetting mode."""
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
    """Builds an 8-element one-hot vector from a metadata row: eye side
    (OS/OD), imaging device (Spectralis/Cirrus 800 FA), and exam year
    (2015-2018). Used as the per-sample "attribute" label for metadata-based
    (not class-based) forgetting."""
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
    """Returns (num_classes, idx_to_class) for the chosen dataset. Class
    counts for the clinical datasets are hardcoded (they're not derivable
    from a generic ImageFolder); svhn/medmnist need special-cased label
    naming, everything else derives idx_to_class from the dataset's own
    class_to_idx mapping."""
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
    elif args.data_name == 'tinyimagenet':
        num_classes = 200
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