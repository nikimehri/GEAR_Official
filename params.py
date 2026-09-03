import argparse
import os


def get_parameters():
    """Defines and parses every CLI flag main.py understands, then runs
    post-parse validation (required-argument combinations, dataset/model
    compatibility, checkpoint paths that must actually exist) before
    returning the parsed args."""
    parser = argparse.ArgumentParser("GEAR Unlearning (contrastive loss + entanglement-score weighting)")

    # which dataset to use
    parser.add_argument('--data_name', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'open_source', 'fundus_3_class', 'oct_4_class', 'oculoplastic',\
     'dr_grade', 'mri', 'ultrasound', 'cxr', 'svhn', 'fashionmnist', 'medmnist'],
                        help='dataset, e.g. cifar10, cifar100, fashionmnist')

    # Which model to use
    parser.add_argument('--model_name', type=str, default='AllCNN', choices=['AllCNN', 'resnet', 'resnet50', 'vit'], help='model name')

    # Model settings
    parser.add_argument('--optim_name', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer name')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='training epoch')


    parser.add_argument('--dataset_dir', type=str, default='./data', help='dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./model_checkpoints',
                        help='checkpoints directory')

    parser.add_argument('--do_unlearning', action='store_true', help='Only unlearning')

    # set train, retrain, or unlearn. Do in separate steps for easier experimentation
    parser.add_argument('--retrain_only', action='store_true', help='retrain dropping a new class')
    parser.add_argument('--train', action='store_true', help='Train model from scratch')

    # Fraction of the training set held out for validation (used for model
    # selection instead of leaking test-set decisions into training).
    parser.add_argument('--val_fraction', type=float, default=0.1, help='fraction of training set to use for validation')

    parser.add_argument('--run_sota', action='store_true', help='run sota method from chen et al')
    parser.add_argument('--specific_settings', action='store_true', help='run unlearning with specific setting ')


    # args for defining what to be unlearned
    parser.add_argument('--forget_class', type=int, default=2, help='forget class')
    parser.add_argument('--custom_unlearn', action='store_true', help='Whether or not to unlearn based on metadata')
    parser.add_argument('--to_forget', type=str, default='Cirrus 800 FA', help='Feature to unlearn from dataset')
    parser.add_argument('--oculoplastics', action='store_true', help='Use oculoplastic dataset')

    # training params
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')


    # model paths for unlearning
    parser.add_argument('--original_model', type=str, help='path to original model')
    parser.add_argument('--retrain_model', type=str, help='path to retrain model')

    parser.add_argument('--gpu_id', type=int,default = 0, help='which GPU to use')
    parser.add_argument('--name', type=str, default = 'placeholder')

    #params for embedding experiments
    parser.add_argument('--tsne_embeddings', action='store_true', help='Only unlearning')
    parser.add_argument('--unlearn_model', type=str, help='path to unlearn model')
    parser.add_argument('--embeddings_name', type=str, help='path to unlearn model')

    parser.add_argument('--remain_reg', type=float, default=0, help='contribution of remain loss to add to total loss')

    parser.add_argument('--good_forget', type=str, help='path to unlearn model with good forget acc')
    parser.add_argument('--good_remain', type=str, help='path to unlearn model with good remain acc')
    parser.add_argument('--ensemble', action='store_true', help='toggle whether or not to do enseble experiments')



    parser.add_argument('--percent_to_forget', type=float,  default =1)
    parser.add_argument('--selective_unlearn', action='store_true', help='toggle whether or not to do selective unlearning experiments')

    # --- GEAR contrastive-loss (CL) arguments ---------------------------------
    # These feed gear.py's feature-space contrastive/alignment losses, added
    # alongside boundary-shrinkage unlearning in Section 4 of the migration.
    parser.add_argument('--feature_contrastive', action='store_true', help='enable feature space contrastive losses during unlearning')
    parser.add_argument('--feature_align_weight', type=float, default=0.0, help='weight for retain to original feature alignment loss')
    parser.add_argument('--retain_forget_weight', type=float, default=0.0, help='weight for retain forget cosine similarity loss')
    parser.add_argument('--forget_forget_weight', type=float, default=0.0, help='weight for forget forget cosine similarity loss')

    # gamma_rep scales the combined contrastive/representation loss (CL+ES);
    # remain_reg (above) separately scales the plain retain cross-entropy loss.
    parser.add_argument('--gamma_rep', type=float, default=1.0,
                        help='weight on contrastive/representation loss. Controls stability regularization strength.')

    parser.add_argument('--target_layer', type=str, default='9',
                        help='Layer for contrastive loss. Use "9" for AllCNN, "layer4" for ResNet-50, '
                             '"all" for multi-layer (layer1+layer2+layer3+layer4) on ResNet-50.')

    # random seed for reproducibility across runs
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility. Set different values for multi-seed experiments.')

    # --- ES (entanglement-score) arguments -----------------------------------
    parser.add_argument('--use_entanglement_weighting', action='store_true',
                        help='Enable entanglement-score weighting for the retain-forget loss (CL+ES). '
                             'Off by default.')
    parser.add_argument('--centroid_refresh_interval', type=int, default=None,
                        help='How many training steps between retain-centroid refreshes. '
                             'Default: one epoch worth of steps (len(train_forget_loader)).')
    parser.add_argument('--cl_warmup_steps', type=int, default=0,
                        help='Steps to linearly ramp gamma_rep from 0 to its target value '
                             'at the start of training. 0 (default) disables the ramp.')

    parser.add_argument('--poison_epoch', type=int, default=10,
                        help='number of epochs for the unlearning poison phase')

    args = parser.parse_args()

    def require_arg(arg_name, condition, reason):
        """Fails fast with a clear message if arg_name is missing whenever
        condition holds, instead of a cryptic error deep inside main()."""
        if condition and not getattr(args, arg_name):
            parser.error(f"--{arg_name} is required when {reason}")

    def require_existing_path(arg_name):
        """Fails fast if a path-valued arg was given but points nowhere."""
        path = getattr(args, arg_name)
        if path and not os.path.exists(path):
            parser.error(f"--{arg_name} points to a missing file: {path}")

    if args.do_unlearning is False:
        if any([args.run_sota, args.specific_settings]):
            raise ValueError("run_sota and specific_settings can only be set if --do_unlearning is true")

    VALID_PAIRINGS = {
        'cifar10':      ['AllCNN'],
        'cifar100':     ['resnet', 'resnet50', 'vit'],
        'fashionmnist': ['AllCNN'],
    }
    if args.data_name in VALID_PAIRINGS:
        allowed = VALID_PAIRINGS[args.data_name]
        if args.model_name not in allowed:
            raise ValueError(
                f"--data_name '{args.data_name}' requires --model_name in {allowed}, "
                f"got '{args.model_name}'"
            )

    require_arg('original_model', args.retrain_only, "--retrain_only is set")
    require_arg('original_model', not args.train and not args.retrain_only and not args.tsne_embeddings,
                "loading checkpoints instead of training from scratch")
    require_arg('retrain_model', not args.train and not args.retrain_only and not args.tsne_embeddings,
                "loading checkpoints instead of training from scratch")

    require_arg('original_model', args.tsne_embeddings, "--tsne_embeddings is set")
    require_arg('retrain_model', args.tsne_embeddings, "--tsne_embeddings is set")
    require_arg('unlearn_model', args.tsne_embeddings, "--tsne_embeddings is set")
    require_arg('embeddings_name', args.tsne_embeddings, "--tsne_embeddings is set")

    for path_arg in ('original_model', 'retrain_model', 'unlearn_model', 'good_forget', 'good_remain'):
        require_existing_path(path_arg)

    return args
