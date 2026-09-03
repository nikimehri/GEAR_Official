"""Pure data module (no functions): checkpoint-path lookup tables used by
baseline_main.py's sweep mode to find each dataset's original/retrain/
baseline-comparison checkpoints. Most entries are commented out - only the
datasets currently being swept are left active."""

import os

# All checkpoint paths below are relative to MODEL_CHECKPOINT_ROOT, which
# defaults to ./model_checkpoints but can be overridden via the
# MODEL_CHECKPOINT_ROOT environment variable to point wherever your
# checkpoints actually live (e.g. a shared drive or a different machine's
# layout).
MODEL_CHECKPOINT_ROOT = os.environ.get('MODEL_CHECKPOINT_ROOT', './model_checkpoints')

model_paths = {
        # 'open_source' : [f'{MODEL_CHECKPOINT_ROOT}/resnet_open_source_fundus/resnetopen_source_original_model_20.pth',
        #                  f'{MODEL_CHECKPOINT_ROOT}/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth',
        #                  'resnet'],

        # 'mri' : [f'{MODEL_CHECKPOINT_ROOT}/resnet_mri/resnetmri_original_model_15.pth',
        #          f'{MODEL_CHECKPOINT_ROOT}/resnet_mri/resnetmri_retrain_model_class_0_15.pth',
        #          'resnet'],

        'cifar10' : [f'{MODEL_CHECKPOINT_ROOT}/allcnn_cifar10/AllCNNcifar10_original_model_25.pth',
                    f'{MODEL_CHECKPOINT_ROOT}/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth',
                    'allcnn'],

        # 'fashionmnist' : [f'{MODEL_CHECKPOINT_ROOT}/allcnn_fashion/AllCNNfashionmnist_original_model_15.pth',
        #                   f'{MODEL_CHECKPOINT_ROOT}/allcnn_fashion/AllCNNfashionmnist_retrain_model_class_0_15.pth',
        #                   'allcnn'],

        # 'medmnist' : [f'{MODEL_CHECKPOINT_ROOT}/allcnn_medmnist/AllCNNmedmnist_original_model_15.pth', 
        #               f'{MODEL_CHECKPOINT_ROOT}/allcnn_medmnist/AllCNNmedmnist_retrain_model_class_0_15.pth',
        #               'allcnn' ]
            }

chen_paths = {
    'cifar10': 'CHEN_UNLEARN_FOR_BASELINE_CIFAR.pth',
    'open_source': 'CHEN_UNLEARN_FOR_BASELINE_CFP.pth',
    'mri': 'CHEN_UNLEARN_FOR_BASELINE_MRI.pth',
    'fashionmnist': 'CHEN_UNLEARN_FOR_BASELINE_FASHION.pth',
    'medmnist': ['CHEN_UNLEARN_FOR_BASELINE_MEDMNIST.pth'],
    'ultrasound' : ['unlearn_ULTRASOUND_CLASS_2_BEST_SOTA.pth'],
    'oct_4_class' : ['unlearn_CLINCAL_OCT_CLASS_0_BEST_SOTA.pth'],
    'oculoplastic' : ['unlearn_OCULOPLASTIC_CLASS_2_BEST_SOTA.pth', 'unlearn_OCULOPLASTIC_CLASS_VPF_BEST_SOTA.pth'],
    'fundus_3_class' : ['unlearn_CLINCAL_CFP_CLASS_0_BEST_SOTA.pth', 'unlearn_CLINCAL_CFP_CAMERA_BEST_SOTA.pth']

}

ravi_paths = {
    'cifar10': 'RAVI_UNLEARN_FOR_BASELINE_CIFAR.pth',
    'open_source': 'RAVI_UNLEARN_FOR_BASELINE_CFP.pth',
    'mri': 'RAVI_UNLEARN_FOR_BASELINE_MRI.pth',
    'fashionmnist': 'RAVI_UNLEARN_FOR_BASELINE_FASHION.pth',
    'medmnist': 'RAVI_UNLEARN_FOR_BASELINE_MEDMNIST.pth' ,
    'ultrasound' : ['unlearn_ULTRASOUND_CLASS_2_BEST.pth'],
    'oct_4_class' : ['unlearn_CLINCAL_OCT_CLASS_0_BEST.pth'],
    'oculoplastic' : ['unlearn_OCULOPLASTIC_CLASS_2_BEST.pth', 'unlearn_OCULOPLASTIC_CLASS_VPF_BEST.pth'],
    'fundus_3_class' : ['unlearn_CLINCAL_CFP_CLASS_0_BEST.pth', 'unlearn_CLINCAL_CFP_CAMERA_BEST.pth']

    }










selective_forget_models = {
        # 'cifar10' :{
        #     'original' : f'{MODEL_CHECKPOINT_ROOT}/allcnn_cifar10/AllCNNcifar10_original_model_25.pth',
        #     'model_type' : 'allcnn',
        #     'chen' :{
        #         # '0.001' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cifar/CHEN_UNLEARN_FOR_BASELINE_CIFAR_001.pth',
        #         '0.01' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cifar/CHEN_UNLEARN_FOR_BASELINE_CIFAR_01.pth',
        #         '0.1' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cifar/CHEN_UNLEARN_FOR_BASELINE_CIFAR_1.pth',
        #         '0.25' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cifar/CHEN_UNLEARN_FOR_BASELINE_CIFAR_25.pth',
        #         '0.5' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cifar/CHEN_UNLEARN_FOR_BASELINE_CIFAR_5.pth',
        #         '0.75' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cifar/CHEN_UNLEARN_FOR_BASELINE_CIFAR_75.pth',
        #         },
        #     'ravi' :{
        #         # '0.001' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cifar/RAVI_UNLEARN_FOR_BASELINE_CIFAR_001.pth',
        #         '0.01' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cifar/RAVI_UNLEARN_FOR_BASELINE_CIFAR_01.pth',
        #         '0.1' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cifar/RAVI_UNLEARN_FOR_BASELINE_CIFAR_1.pth',
        #         '0.25' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cifar/RAVI_UNLEARN_FOR_BASELINE_CIFAR_25.pth',
        #         '0.5' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cifar/RAVI_UNLEARN_FOR_BASELINE_CIFAR_5.pth',
        #         '0.75' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cifar/RAVI_UNLEARN_FOR_BASELINE_CIFAR_75.pth',
        #         },
        #     'retrain' :{
        #         # '0.001' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/AllCNN_cifar10_retrain_CHEN_UNLEARN_FOR_BASELINE_CIFAR_001_25.pth',
        #         '0.01' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_cifar/AllCNN_cifar10__retrain_CHEN_UNLEARN_FOR_BASELINE_CIFAR_01_25.pth',
        #         '0.1' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_cifar/AllCNN_cifar10_retrain_CHEN_UNLEARN_FOR_BASELINE_CIFAR_1_25.pth',
        #         '0.25' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_cifar/AllCNN_cifar10_retrain_CHEN_UNLEARN_FOR_BASELINE_CIFAR_25_25.pth',
        #         '0.5' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_cifar/AllCNN_cifar10_retrain_CHEN_UNLEARN_FOR_BASELINE_CIFAR_5_25.pth',
        #         '0.75' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_cifar/AllCNN_cifar10_retrain_CHEN_UNLEARN_FOR_BASELINE_CIFAR_75_25.pth',
        #         }}
        #                 ,
        # 'open_source' :{
        #     'original' : f'{MODEL_CHECKPOINT_ROOT}/resnet_open_source_fundus/resnetopen_source_original_model_20.pth',
        #     'model_type' : 'resnet',
        #     'chen' :{
        #         # '.001' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cfp/CHEN_UNLEARN_FOR_BASELINE_CFP_OS_001.pth',
        #         # '0.01' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cfp/CHEN_UNLEARN_FOR_BASELINE_CFP_OS_01.pth',
        #         '0.1' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cfp/CHEN_UNLEARN_FOR_BASELINE_CFP_OS_1.pth',
        #         '0.25' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cfp/CHEN_UNLEARN_FOR_BASELINE_CFP_OS_25.pth',
        #         '0.5' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cfp/CHEN_UNLEARN_FOR_BASELINE_CFP_OS_5.pth',
        #         '0.75' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cfp/CHEN_UNLEARN_FOR_BASELINE_CFP_OS_75.pth',
        #         },
        #     'ravi' :{
        #         # '.001' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cfp/RAVI_UNLEARN_FOR_BASELINE_CFP_OS_001.pth',
        #         # '0.01' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cfp/RAVI_UNLEARN_FOR_BASELINE_CFP_OS_01.pth',
        #         '0.1' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cfp/RAVI_UNLEARN_FOR_BASELINE_CFP_OS_1.pth',
        #         '0.25' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cfp/RAVI_UNLEARN_FOR_BASELINE_CFP_OS_25.pth',
        #         '0.5' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cfp/RAVI_UNLEARN_FOR_BASELINE_CFP_OS_5.pth',
        #         '0.75' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cfp/RAVI_UNLEARN_FOR_BASELINE_CFP_OS_75.pth',
        #         } ,
        #     'retrain' :{
        #         # '.001' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_cfp_os/AllCNN_cifar10_retrain_CHEN_UNLEARN_FOR_BASELINE_CIFAR_001_25.pth',
        #         # '0.01' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_cfp/resnet_open_source_retrain_CHEN_UNLEARN_FOR_BASELINE_CFP_OS_01_15.pth',
        #         '0.1' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_cfp/resnet_open_source_retrain_CHEN_UNLEARN_FOR_BASELINE_CFP_OS_1_15.pth',
        #         '0.25' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_cfp/resnet_open_source_retrain_CHEN_UNLEARN_FOR_BASELINE_CFP_OS_25_15.pth',
        #         '0.5' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_cfp/resnet_open_source_retrain_CHEN_UNLEARN_FOR_BASELINE_CFP_OS_5_15.pth',
        #         '0.75' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_cfp/resnet_open_source_retrain_CHEN_UNLEARN_FOR_BASELINE_CFP_OS_75_15.pth',
        #         }
        #     },


        'fashionmnist' :{
            'original' : f'{MODEL_CHECKPOINT_ROOT}/allcnn_fashion/AllCNNfashionmnist_original_model_15.pth',
            'model_type' : 'allcnn',
            'chen' :{
                # '0.001' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cifar/CHEN_UNLEARN_FOR_BASELINE_CIFAR_001.pth',
                '0.01' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_fashion/CHEN_UNLEARN_FOR_BASELINE_FASHION_01.pth',
                '0.1' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_fashion/CHEN_UNLEARN_FOR_BASELINE_FASHION_1.pth',
                '0.25' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_fashion/CHEN_UNLEARN_FOR_BASELINE_FASHION_25.pth',
                '0.5' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_fashion/CHEN_UNLEARN_FOR_BASELINE_FASHION_5.pth',
                '0.75' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_fashion/CHEN_UNLEARN_FOR_BASELINE_FASHION_75.pth',
                },
            'ravi' :{
                # '0.001' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cifar/RAVI_UNLEARN_FOR_BASELINE_CIFAR_001.pth',
                '0.01' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_fashion/RAVI_UNLEARN_FOR_BASELINE_FASHION_01.pth',
                '0.1' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_fashion/RAVI_UNLEARN_FOR_BASELINE_FASHION_1.pth',
                '0.25' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_fashion/RAVI_UNLEARN_FOR_BASELINE_FASHION_25.pth',
                '0.5' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_fashion/RAVI_UNLEARN_FOR_BASELINE_FASHION_5.pth',
                '0.75' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_fashion/RAVI_UNLEARN_FOR_BASELINE_FASHION_75.pth',
                },
            'retrain' :{
                # '0.001' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/AllCNN_cifar10_retrain_CHEN_UNLEARN_FOR_BASELINE_CIFAR_001_25.pth',
                '0.01' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_fashion/AllCNN_fashionmnist_retrain_CHEN_UNLEARN_FOR_BASELINE_FASHION_01_15.pth',
                '0.1' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_fashion/AllCNN_fashionmnist_retrain_CHEN_UNLEARN_FOR_BASELINE_FASHION_1_15.pth',
                '0.25' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_fashion/AllCNN_fashionmnist_retrain_CHEN_UNLEARN_FOR_BASELINE_FASHION_25_15.pth',
                '0.5' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_fashion/AllCNN_fashionmnist_retrain_CHEN_UNLEARN_FOR_BASELINE_FASHION_5_15.pth',
                '0.75' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_fashion/AllCNN_fashionmnist_retrain_CHEN_UNLEARN_FOR_BASELINE_FASHION_75_15.pth',
                }}
                        ,
        'mri' :{
            'original' : f'{MODEL_CHECKPOINT_ROOT}/resnet_mri/resnetmri_original_model_15.pth',
            'model_type' : 'resnet',
            'chen' :{
                # '.001' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cfp/CHEN_UNLEARN_FOR_BASELINE_CFP_OS_001.pth',
                '0.01' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_MRI/CHEN_UNLEARN_FOR_BASELINE_MRI_01.pth',
                '0.1' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_MRI/CHEN_UNLEARN_FOR_BASELINE_MRI_1.pth',
                '0.25' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_MRI/CHEN_UNLEARN_FOR_BASELINE_MRI_25.pth',
                '0.5' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_MRI/CHEN_UNLEARN_FOR_BASELINE_MRI_5.pth',
                '0.75' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_MRI/CHEN_UNLEARN_FOR_BASELINE_MRI_75.pth',
                },
            'ravi' :{
                # '.001' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_cfp/CHEN_UNLEARN_FOR_BASELINE_CFP_OS_001.pth',
                '0.01' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_MRI/RAVI_UNLEARN_FOR_BASELINE_MRI_01.pth',
                '0.1' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_MRI/RAVI_UNLEARN_FOR_BASELINE_MRI_1.pth',
                '0.25' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_MRI/RAVI_UNLEARN_FOR_BASELINE_MRI_25.pth',
                '0.5' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_MRI/RAVI_UNLEARN_FOR_BASELINE_MRI_5.pth',
                '0.75' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_MRI/RAVI_UNLEARN_FOR_BASELINE_MRI_75.pth',
                } ,
            'retrain' :{
                # '.001' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/specific_unlearn_MRI/RAVI_UNLEARN_FOR_BASELINE_MRI_01.pth',
                '0.01' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_MRI/resnet_mri_retrain_CHEN_UNLEARN_FOR_BASELINE_MRI_01_15.pth',
                '0.1' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_MRI/resnet_mri_retrain_CHEN_UNLEARN_FOR_BASELINE_MRI_1_15.pth',
                '0.25' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_MRI/resnet_mri_retrain_CHEN_UNLEARN_FOR_BASELINE_MRI_25_15.pth',
                '0.5' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_MRI/resnet_mri_retrain_CHEN_UNLEARN_FOR_BASELINE_MRI_5_15.pth',
                '0.75' : f'{MODEL_CHECKPOINT_ROOT}/baseline_models/retrain_specific_MRI/resnet_mri_retrain_CHEN_UNLEARN_FOR_BASELINE_MRI_75_15.pth',
                }
            }
}


med_unlearn_paths = {
        # 'ultrasound' : [f'{MODEL_CHECKPOINT_ROOT}/resnet_ultrasound/resnetultrasound_original_model_30.pth',
        #                 f'{MODEL_CHECKPOINT_ROOT}/resnet_ultrasound/resnetultrasound_retrain_model_class_0_30.pth',
        #                 'resnet',
        #                 '0',
        #                 False],
        # 'ultrasound' : [f'{MODEL_CHECKPOINT_ROOT}/resnet_ultrasound/resnetultrasound_original_model_30.pth',
        #                 f'{MODEL_CHECKPOINT_ROOT}/resnet_ultrasound/resnetultrasound_retrain_model_class_2_30.pth',
        #                 'resnet',
        #                 '2',
        #                 False],
        # 'oct_4_class' : [f'{MODEL_CHECKPOINT_ROOT}/resnet_open_source_oct/resnetoct_4_class_original_model_20.pth',
        #                 f'{MODEL_CHECKPOINT_ROOT}/resnet_open_source_oct/resnet_oct_4_class_retrain_unlearn_CLINCAL_OCT_CLASS_0_LAMDA_0_RETRAIN_14.pth',
        #                 'resnet',
        #                 '0',
        #                 False],
        'oculoplastic' : [f'{MODEL_CHECKPOINT_ROOT}/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth',
                          f'{MODEL_CHECKPOINT_ROOT}/resnet_oculoplastic/resnet_oculoplastic_retrain_TRAIN_OCULOPLASTIC_CLASS_2_26.pth',
                          'resnet',
                          '2',
                          False],
        # 'oculoplastic' : [f'{MODEL_CHECKPOINT_ROOT}/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth',
        #                   f'{MODEL_CHECKPOINT_ROOT}/resnet_oculoplastic/resnet_oculoplastic_retrain_TRAIN_OCULOPLASTIC_CLASS_0_26.pth',
        #                   'resnet',
        #                   '0',
        #                   False],
        # 'oculoplastic' : [f'{MODEL_CHECKPOINT_ROOT}/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth',
        #                   f'{MODEL_CHECKPOINT_ROOT}/resnet_oculoplastic/resnet_oculoplastic_retrain_retrain_OCULOPLASTIC_CLASS_VPF_29.pth',
        #                   'resnet',
        #                   '0',
        #                   True],
        'fundus_3_class' : [f'{MODEL_CHECKPOINT_ROOT}/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth',
                            f'{MODEL_CHECKPOINT_ROOT}/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_40_16_bs_class_0.pth',
                          'resnet',
                            '0',
                            False]
                            ,
        # 'fundus_3_class' : [f'{MODEL_CHECKPOINT_ROOT}/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth',
        #                     f'{MODEL_CHECKPOINT_ROOT}/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_45_16_bs_class_cirrus.pth',
        #                   'resnet',
        #                     '2',
        #                     True]
                            }