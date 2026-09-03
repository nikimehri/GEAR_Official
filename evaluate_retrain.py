
# This is to evaluate retrain checkpoint on test_remain_loader.
import argparse
import torch
from torch.utils.data import random_split

from make_dataloaders import dataloader_engine, get_dataset
from trainer import load_checkpoint, eval as eval_model
from utils import set_num_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--data_name', default='cifar100')
    parser.add_argument('--dataset_dir', default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--forget_class', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--custom_unlearn', action='store_true')
    parser.add_argument('--oculoplastics', action='store_true')
    parser.add_argument('--selective_unlearn', action='store_true')
    parser.add_argument('--percent_to_forget', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    )

    trainset_full, testset, dataset = get_dataset(args.data_name, args.dataset_dir)

    val_size = int(len(trainset_full) * 0.1)
    train_size = len(trainset_full) - val_size
    trainset, valset = random_split(
        trainset_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    num_classes, idx_to_class = set_num_classes(args, dataset)
    total_forget_class = sum(1 for _, t in trainset if t == args.forget_class)
    num_forget = int(total_forget_class * args.percent_to_forget)

    _, _, _, _, _, test_remain_loader, \
    _, _, _, _, _, _, \
    _, _, _ = dataloader_engine(
        args, trainset, valset, testset,
        combined_df=None,
        num_forget=num_forget,
        oculoplastics=args.oculoplastics,
        selective_unlearning=args.selective_unlearn
    )

    model = load_checkpoint(args.checkpoint_path, 'retrain_model', device=device)

    _, remain_acc = eval_model(model, test_remain_loader, mode='', device=device)

    print(f'Checkpoint: {args.checkpoint_path}')
    print(f'Seed: {args.seed}')
    print(f'remain_acc: {remain_acc:.4f}')


if __name__ == '__main__':
    main()