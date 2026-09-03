

###################### FUNDUS



python main.py --forget_class 0 \
               --data_name fundus_3_class \
               --model_name resnet \
               --lr 0.001 \
                --epoch 40  \
               --dataset_dir ./data/fundus_big \
               --batch_size 16 \
               --gpu_id 0 \
               --run_sota \
               --do_unlearning \
               --name  'unlearn_CLINCAL_CFP_CLASS_0_LAMDA_0001_RUN_SOTA' \
                --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
                --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_40_16_bs_class_0.pth' \
                --remain_reg 0.0 \
                --lamda .0001


# python main.py --forget_class 0 \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --lr 0.001 \
#                 --epoch 40  \
#                --dataset_dir ./data/fundus_big \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_CFP_CLASS_0_LAMDA_0001' \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_40_16_bs_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .0001


# python main.py --forget_class 0 \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --lr 0.001 \
#                 --epoch 40  \
#                --dataset_dir ./data/fundus_big \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_CFP_CLASS_0_LAMDA_001' \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_40_16_bs_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .001


# python main.py --forget_class 0 \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --lr 0.001 \
#                 --epoch 40  \
#                --dataset_dir ./data/fundus_big \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_CFP_CLASS_0_LAMDA_01' \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_40_16_bs_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .01


# python main.py --forget_class 0 \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --lr 0.001 \
#                 --epoch 40  \
#                --dataset_dir ./data/fundus_big \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_CFP_CLASS_0_LAMDA_1' \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_40_16_bs_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .1


# python main.py --forget_class 0 \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --lr 0.001 \
#                 --epoch 40  \
#                --dataset_dir ./data/fundus_big \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_CFP_CLASS_0_LAMDA_0' \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_40_16_bs_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --lamda 0