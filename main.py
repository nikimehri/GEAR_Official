
import argparse
import numpy as np
import boundary_unlearning
from utils import *
from trainer import *
import shutil
import os 
import pandas as pd
import time
import csv
import json
import torch
import torch.nn as nn
import torch.cuda
from params import get_parameters
from make_dataloaders import *
from embeddings import *
from ensemble import run_exps


def main(args):
    torch.cuda.empty_cache()
    seed_torch()
        # TODO improve logging   
    # gamma_values = [.1, 0.5, 0.9]
    gamma_values = [0, .0001, .1, 1]

    l1_norms = [False]
    # distributions = ['normal', 'cauchy', 'laplacian', 'uniform']
    distributions = ['normal']

    csv_columns, output_file_name = set_up_save(args, distributions, gamma_values, l1_norms, args.name)

    # set device 
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    create_dir(args.dataset_dir)
    create_dir(args.checkpoint_dir)

    path = args.checkpoint_dir + '/'
    model_name = path[2:] + args.model_name + '_' + args.data_name 

    combined_df = None

    if args.custom_unlearn:
        ord_df = pd.read_csv('data/csvs_fundus/Other_Retinal_Disorders_UNIQUE_MRN_filtered.csv')
        dr_df = pd.read_csv('data/csvs_fundus/Diabetic_Retinopathy_UNIQUE_MRN_filtered.csv')
        glauc_df = pd.read_csv('data/csvs_fundus/Glaucoma_UNIQUE_MRN_filtered.csv')
        combined_df = pd.concat([ord_df, dr_df, glauc_df], ignore_index=True)
    
    if args.custom_unlearn and args.oculoplastics:
        ted_df = pd.read_csv('data/csvs_oculoplastic/mm_07022024_full_run_TED_GT_pix.csv')
        cfd_df = pd.read_csv('data/csvs_oculoplastic/mm_07022024_full_run_CFD_GT_pix.csv')
        combined_df = pd.concat([ted_df, cfd_df], ignore_index=True)
        print(combined_df.head())
    
    SELECTIVE_UNLEARNING = args.selective_unlearn

    if SELECTIVE_UNLEARNING == False:
        FORGET_PERCENTAGE = 1
    else:
        print('💀BEWARE: MAKE SURE YOU REALLY WANT TO USE SELECTIVE UNLEARNING💀')
        FORGET_PERCENTAGE = args.percent_to_forget



    trainset_full, testset, dataset = get_dataset(args.data_name, args.dataset_dir)

    # Hold out a validation split from the training set (seeded for
    # reproducibility) so model selection can happen against data the model
    # never trains on, instead of leaking test-set decisions into training.
    val_size = int(len(trainset_full) * args.val_fraction)
    train_size = len(trainset_full) - val_size
    split_generator = torch.Generator().manual_seed(args.seed)
    trainset, valset = torch.utils.data.random_split(
        trainset_full, [train_size, val_size], generator=split_generator
    )

    train_loader, test_loader = get_dataloader(trainset, testset, args.batch_size, device=device)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True)

    # set number of classes
    num_classes, idx_to_class = set_num_classes(args, dataset)
    total_forget_class = sum(1 for _, target in trainset if target == args.forget_class)

    num_forget = int(total_forget_class * FORGET_PERCENTAGE)
    print(f"Number to forget: {num_forget}")


    train_forget_loader, train_remain_loader, \
        val_forget_loader, val_remain_loader, \
        test_forget_loader, test_remain_loader, \
        _, _, _, _, _, _, \
        train_dict, val_dict, test_dict = dataloader_engine(
            args, trainset, valset, testset, combined_df, num_forget=num_forget,
            oculoplastics=args.oculoplastics,
            selective_unlearning=SELECTIVE_UNLEARNING
        )
    
 

    if args.tsne_embeddings:
        print('doing embeddings')

        fig, axes = plt.subplots(1, 3, figsize=(18, 6)) 

        original_model = torch.load(args.original_model)
        retrained_model = torch.load(args.retrain_model)
        unlearned_model = torch.load(args.unlearn_model)
        # save_unlearning_examples(unlearned_model, test_forget_loader, test_remain_loader, device, save_dir="example_images")

        models = [original_model, retrained_model, unlearned_model]
        titles = ['Original Model', 'Retrained Model', 'Unlearned Model']

        for i, (ax, model, title) in enumerate(zip(axes, models, titles)):
            model.to(device)
            
            embeddings, predictions, is_forget_sample, true_labels = get_embeddings_predictions_and_forget_indications(
                model, test_forget_loader, test_remain_loader, device
            )
            
            plot_tsne(embeddings, predictions, is_forget_sample, true_labels, title, ax, add_legend=False)#(i == 0))
        
        plt.tight_layout()
        plt.savefig(args.embeddings_name + '.png', dpi=600)
        
        breakpoint


    '''
    Not included in TMLR paper but keeping for future work
    '''
    if args.ensemble:
        run_exps(args, testset, trainset, train_remain_loader, finetune=True, frozen=False)


    ori_model, retrain_model, row_data = train_engine(args, train_remain_loader, val_remain_loader, train_loader, val_loader,
                 dataset, num_classes, idx_to_class, device, model_name, output_file_name,
                 csv_columns, distributions, gamma_values,exp_name = args.name)

    if args.do_unlearning:
        '''
        set gamma and lambda = 0
        '''
        if args.run_sota:
            print('DOING BOUNDARY SHRINKAGE WITH SOTA METHOD')
            save_me = args.name + '_SOTA'
            unlearn_model_sota, forget_acc_sota, remain_acc_sota, unlearning_time = boundary_unlearning.boundary_shrink(
                    ori_model, train_forget_loader, trainset, testset, test_loader, device, 
                    forget_class=args.forget_class, path=path, custom_forget=args.custom_unlearn, to_forget=args.to_forget,
                    test_metadata=test_dict, train_metadata=train_dict,gamma=0, dist='normal',
                    output_name=save_me, use_linfpgd=args.use_linfpgd, lamda=0, l1_norm=False, data_name = args.data_name, scaling = None, 
                    oculoplastics=args.oculoplastics, retrain_model=retrain_model, train_remain_loader=train_remain_loader, use_logits = args.use_logits,
                    remain_reg_param= args.remain_reg, logit_preprocess= args.logit_preprocess, selective_unlearning=SELECTIVE_UNLEARNING
            )

            # Calculate per class accuracy
            per_class_accs_sota = test(unlearn_model_sota, test_loader, idx_to_class, num_classes, device)

            print(f'SOTA UNLEARNING TIME forgetting {num_forget} SAMPLES: {unlearning_time}')

            # Update the CSV file with the SOTA results
            with open(output_file_name, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                row_data['Forget Acc SOTA'] = forget_acc_sota.detach().item()
                row_data['Remain Acc SOTA'] = remain_acc_sota.detach().item()
                row_data['Unlearning Time'] = unlearning_time
                row_data['Per Class Accuracies SOTA'] = json.dumps(per_class_accs_sota)
                writer.writerow(row_data)


    
        '''
        Use argument specific_settings to unlearn using specific settings
        '''
        if args.specific_settings:
            lamda = args.lamda
            gamma = args.gamma
            dist = 'normal'
            save_me = args.name

            unlearn_model, forget_acc, remain_acc, boundary_shrink_time = boundary_unlearning.boundary_shrink(
                ori_model, train_forget_loader, trainset, testset, test_loader, device,
                forget_class=args.forget_class, path=path, custom_forget=args.custom_unlearn, to_forget=args.to_forget,
                test_metadata=test_dict, train_metadata=train_dict, gamma=args.gamma,
                dist=dist, output_name=save_me, use_linfpgd=False, lamda=args.lamda, l1_norm=False, data_name = args.data_name,scaling = None, 
                oculoplastics=args.oculoplastics, retrain_model=retrain_model, train_remain_loader=train_remain_loader, use_logits = args.use_logits,
                        remain_reg_param= args.remain_reg, logit_preprocess= args.logit_preprocess, selective_unlearning=SELECTIVE_UNLEARNING
            )
            unlearn_model.to(device)

            print(f'MODIFIED UNLEARNING TIME for gamma {args.gamma} lambda {args.lamda} forgetting {num_forget} SAMPLES: {boundary_shrink_time} ')

            per_class_accs = test(unlearn_model, test_loader, idx_to_class, num_classes, device)

            # Update row data
            row_data[f'Forget Acc {dist}_lambda_{gamma}'] = forget_acc.detach().item()
            row_data[f'Remain Acc {dist}_lambda_{gamma}'] = remain_acc.detach().item()
            row_data[f'Per Class Accuracies {dist}_lambda_{gamma}'] = json.dumps(per_class_accs)

            with open(output_file_name, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writerow(row_data)



if __name__ == '__main__':
    args = get_parameters()
    main(args)



















