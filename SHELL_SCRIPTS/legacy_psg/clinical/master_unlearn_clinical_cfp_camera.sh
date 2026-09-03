

###################### FUNDUS camera




python main.py --custom_unlearn \
                --to_forget 'Cirrus 800 FA' \
               --data_name fundus_3_class \
               --model_name resnet \
               --lr 0.001 \
                --epoch 40  \
               --dataset_dir ./data/fundus_big \
               --batch_size 16 \
               --gpu_id 0 \
               --sgld \
               --do_unlearning \
               --name  'unlearn_CLINCAL_CFP_CLASS_camera_LAMDA_0' \
                --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
                --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_45_16_bs_class_cirrus.pth' \
                --remain_reg 0.0 \
                --lamda 0





# python main.py --custom_unlearn \
#                 --to_forget 'Cirrus 800 FA' \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --lr 0.001 \
#                 --epoch 40  \
#                --dataset_dir ./data/fundus_big \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_CFP_CLASS_camera_LAMDA_0001' \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_45_16_bs_class_cirrus.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .0001


# python main.py --custom_unlearn \
#                 --to_forget 'Cirrus 800 FA' \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --lr 0.001 \
#                 --epoch 40  \
#                --dataset_dir ./data/fundus_big \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_CFP_CLASS_camera_LAMDA_001' \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_45_16_bs_class_cirrus.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .001


# python main.py --custom_unlearn \
#                 --to_forget 'Cirrus 800 FA' \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --lr 0.001 \
#                 --epoch 40  \
#                --dataset_dir ./data/fundus_big \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_CFP_CLASS_camera_LAMDA_01' \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_45_16_bs_class_cirrus.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .01


# python main.py --custom_unlearn \
#                 --to_forget 'Cirrus 800 FA' \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --lr 0.001 \
#                 --epoch 40  \
#                --dataset_dir ./data/fundus_big \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_CFP_CLASS_camera_LAMDA_1' \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_45_16_bs_class_cirrus.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .1

