"""One-time setup script for the TinyImageNet-200 dataset.

Downloads and extracts tiny-imagenet-200.zip (if not already present under
--dataset_dir), then reorganizes its val/ split into per-class subdirectories
so make_dataloaders.get_dataset('tinyimagenet', ...) can load both splits via
ImageFolder the same way. train/ already ships in ImageFolder-compatible
per-class subdirectories; val/ ships flat (all images in one directory) with
a separate val_annotations.txt mapping filename -> class, which this script
resolves.

Usage:
    python scripts/prepare_tinyimagenet.py --dataset_dir ./data
"""
import argparse
import os
import shutil
import urllib.request
import zipfile

TINYIMAGENET_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'


def download_and_extract(dataset_dir):
    """Downloads+extracts the dataset zip if tiny-imagenet-200/ doesn't
    already exist under dataset_dir. Returns the extracted root path."""
    root = os.path.join(dataset_dir, 'tiny-imagenet-200')
    if os.path.isdir(root):
        print(f'{root} already exists, skipping download/extract.')
        return root

    os.makedirs(dataset_dir, exist_ok=True)
    zip_path = os.path.join(dataset_dir, 'tiny-imagenet-200.zip')
    if not os.path.exists(zip_path):
        print(f'Downloading TinyImageNet-200 from {TINYIMAGENET_URL} ...')
        urllib.request.urlretrieve(TINYIMAGENET_URL, zip_path)

    print(f'Extracting {zip_path} ...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dataset_dir)

    return root


def reorganize_val(root):
    """Moves val/images/*.JPEG into val/<class_id>/*.JPEG per
    val_annotations.txt, matching train/'s per-class subdirectory layout.
    Idempotent: if val/images no longer exists, a prior run already did this."""
    val_dir = os.path.join(root, 'val')
    images_dir = os.path.join(val_dir, 'images')
    annotations_path = os.path.join(val_dir, 'val_annotations.txt')

    if not os.path.isdir(images_dir):
        print(f'{images_dir} not found - val/ already reorganized, skipping.')
        return

    filename_to_class = {}
    with open(annotations_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            filename_to_class[parts[0]] = parts[1]

    for filename, class_id in filename_to_class.items():
        class_dir = os.path.join(val_dir, class_id)
        os.makedirs(class_dir, exist_ok=True)
        src = os.path.join(images_dir, filename)
        if os.path.exists(src):
            shutil.move(src, os.path.join(class_dir, filename))

    shutil.rmtree(images_dir)
    print(f'Reorganized {len(filename_to_class)} validation images into per-class subdirectories.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and prepare TinyImageNet-200 for ImageFolder loading.')
    parser.add_argument('--dataset_dir', type=str, default='./data',
                        help='directory tiny-imagenet-200/ lives under (or will be downloaded/extracted into)')
    args = parser.parse_args()

    dataset_root = download_and_extract(args.dataset_dir)
    reorganize_val(dataset_root)
    print(f'TinyImageNet-200 ready at {dataset_root}')
