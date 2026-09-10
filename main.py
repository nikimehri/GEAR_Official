
import gear
from utils import *
from trainer import *
import pandas as pd
import csv
import json
import torch
import torch.cuda
from params import get_parameters
from make_dataloaders import *
from embeddings import *
from ensemble import run_exps


def main(args):
    """Top-level orchestration: build the dataset/dataloaders and train/val/
    test split, optionally run t-SNE embedding analysis or the ensemble
    side-experiment, get an original+retrain model pair (training from
    scratch, retraining only, or loading existing checkpoints), and - if
    --do_unlearning - run GEAR unlearning once against the test set
    (--run_sota) and/or once against the validation set (--specific_settings),
    logging results to CSV after each stage."""
    torch.cuda.empty_cache()
    # Was seed_torch() with no argument, which always used seed_torch's own
    # default (2022) regardless of --seed - meaning weight init, dropout,
    # augmentation, and GEAR's own training stochasticity never actually
    # varied across --seed values, only the train/val split did (that part
    # uses its own independent torch.Generator, seeded from args.seed
    # separately, below). Passing args.seed here is what makes --seed
    # actually control every source of randomness, not just the split.
    seed_torch(args.seed)

    csv_columns, output_file_name = set_up_save(args, args.name)
    # Shared across every sweep invocation of this script (e.g. multiple
    # hyperparameter runs with the same --name) so each run appends one row.
    results_csv = f"{args.name}_sweep_results.csv"

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



    trainset_full, testset, dataset = get_dataset(args.data_name, args.dataset_dir, model_name=args.model_name)

    # Hold out a validation split from the training set (seeded for
    # reproducibility) so model selection can happen against data the model
    # never trains on, instead of leaking test-set decisions into training.
    val_size = int(len(trainset_full) * args.val_fraction)
    train_size = len(trainset_full) - val_size
    split_generator = torch.Generator().manual_seed(args.seed)
    trainset, valset = torch.utils.data.random_split(
        trainset_full, [train_size, val_size], generator=split_generator
    )

    train_loader, test_loader = get_dataloader(trainset, testset, args.batch_size)
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


    
    if args.ensemble:
        run_exps(args, valset, testset, train_remain_loader, finetune=True, frozen=False)


    ori_model, retrain_model, row_data = train_engine(args, train_remain_loader, val_remain_loader, train_loader, val_loader,
                 dataset, num_classes, idx_to_class, device, model_name, output_file_name,
                 csv_columns, exp_name = args.name)

    if args.do_unlearning:
        # Shared GEAR keyword arguments for both modes below - the only
        # differences between --run_sota and --specific_settings are which
        # split gets evaluated (test vs. validation) and the output name.
        gear_kwargs = dict(
            forget_class=args.forget_class, custom_forget=args.custom_unlearn, to_forget=args.to_forget,
            oculoplastics=args.oculoplastics,
            retrain_model=retrain_model, train_remain_loader=train_remain_loader,
            remain_reg_param=args.remain_reg, selective_unlearning=SELECTIVE_UNLEARNING,
            poison_epoch=args.poison_epoch,
            feature_contrastive=args.feature_contrastive,
            feature_align_weight=args.feature_align_weight,
            retain_forget_weight=args.retain_forget_weight,
            forget_forget_weight=args.forget_forget_weight,
            gamma_rep=args.gamma_rep,
            target_layer=args.target_layer,
            results_csv=results_csv,
            use_entanglement_weighting=args.use_entanglement_weighting,
            centroid_refresh_interval=args.centroid_refresh_interval,
            centroid_mode=args.centroid_mode,
            num_classes=num_classes,
            cl_warmup_steps=args.cl_warmup_steps,
            data_name=args.data_name,
            compute_ain=args.compute_ain,
            ain_error_range=args.ain_error_range,
            ain_lr=args.ain_lr,
            ain_max_epochs=args.ain_max_epochs,
            ain_eval_interval=args.ain_eval_interval,
            seed=args.seed,
        )

        if args.run_sota:
            print('RUNNING GEAR UNLEARNING (evaluated against the test set)')
            save_me = args.name + '_SOTA'
            unlearn_model_sota, forget_acc_sota, remain_acc_sota, unlearning_time, test_acc_sota, mia_score_sota, \
                retain_adjacent_acc_sota, retain_remote_acc_sota, ain_score_sota = gear.gear(
                ori_model, train_forget_loader, trainset, testset, test_loader, device,
                test_metadata=test_dict, train_metadata=train_dict, output_name=save_me,
                **gear_kwargs
            )

            # Calculate per class accuracy
            per_class_accs_sota = test(unlearn_model_sota, test_loader, idx_to_class, num_classes, device)

            print(f'SOTA UNLEARNING TIME forgetting {num_forget} SAMPLES: {unlearning_time}')

            # Update the CSV file with the SOTA results. Retain Remote/Adjacent
            # Acc, Test Acc, MIA, and AIN share column names with the
            # --specific_settings row below (not suffixed "SOTA") - each mode
            # writes its own row immediately after computing these values, so
            # the two rows never cross-contaminate; which row is which is
            # distinguishable via 'Forget Acc SOTA' vs. 'Forget Acc' being set.
            with open(output_file_name, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                row_data['Forget Acc SOTA'] = forget_acc_sota.detach().item()
                row_data['Remain Acc SOTA'] = remain_acc_sota.detach().item()
                row_data['Unlearning Time'] = unlearning_time
                row_data['Per Class Accuracies SOTA'] = json.dumps(per_class_accs_sota)
                row_data['Retain Remote Acc'] = retain_remote_acc_sota
                row_data['Retain Adjacent Acc'] = retain_adjacent_acc_sota
                row_data['Test Acc'] = test_acc_sota.detach().item() if isinstance(test_acc_sota, torch.Tensor) else test_acc_sota
                row_data['MIA'] = mia_score_sota
                row_data['AIN'] = ain_score_sota
                writer.writerow(row_data)


        '''
        Use argument specific_settings to unlearn evaluated against the
        held-out validation set (rather than test), matching the model
        selection convention used elsewhere in the pipeline.
        '''
        if args.specific_settings:
            save_me = args.name

            unlearn_model, forget_acc, remain_acc, gear_time, test_acc, mia_score, retain_adjacent_acc, retain_remote_acc, ain_score = gear.gear(
                ori_model, train_forget_loader, trainset, valset, val_loader, device,
                test_metadata=val_dict, train_metadata=train_dict, output_name=save_me,
                **gear_kwargs
            )
            unlearn_model.to(device)

            print(f'GEAR UNLEARNING TIME forgetting {num_forget} SAMPLES: {gear_time}')

            # Fixed columns for the GEAR-specific run. Retain Remote/Adjacent
            # Acc are 'N/A' for any dataset without a known class hierarchy
            # (see class_hierarchy.py) - currently only cifar100/tinyimagenet.
            row_data['Forget Acc'] = forget_acc.detach().item()
            row_data['Retain Remote Acc'] = retain_remote_acc
            row_data['Retain Adjacent Acc'] = retain_adjacent_acc
            row_data['Test Acc'] = test_acc.detach().item() if isinstance(test_acc, torch.Tensor) else test_acc
            row_data['MIA'] = mia_score
            # 'N/A' unless --compute_ain was set (gear() already returns 'N/A'
            # by default - AIN is opt-in since it involves actual retraining).
            row_data['AIN'] = ain_score
            row_data['Unlearning Time'] = gear_time

            with open(output_file_name, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writerow(row_data)



if __name__ == '__main__':
    args = get_parameters()
    main(args)



















