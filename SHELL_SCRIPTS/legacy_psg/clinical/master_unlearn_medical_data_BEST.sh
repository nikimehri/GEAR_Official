###################### FUNDUS

# ## Class 0

# python main.py --forget_class 0 \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --lr 0.001 \
#                 --epoch 40  \
#                --dataset_dir ./data/fundus_big \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_CFP_CLASS_0_BEST' \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_40_16_bs_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --run_sota \
#                 --specific_settings \
#                 --gamma .0001 \
#                 --lamda 0


# python main.py --custom_unlearn \
#                 --to_forget 'Cirrus 800 FA' \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --lr 0.001 \
#                 --epoch 40  \
#                --dataset_dir ./data/fundus_big \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_CFP_CAMERA_BEST' \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_45_16_bs_class_cirrus.pth' \
#                 --remain_reg 0.0 \
#                 --run_sota \
#                 --specific_settings \
#                 --gamma .0001 \
#                 --lamda 0



# ###################### OCT 

# ## CLASS 0 

# python main.py --forget_class 0 \
#                --data_name oct_4_class \
#                --model_name resnet \
#               --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/oct_open_source \
#                --batch_size 8 \
#                --gpu_id 0 \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_OCT_CLASS_0_BEST' \
#                 --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_20.pth' \
#                 --retrain_model  'model_checkpoints/resnet_open_source_oct/resnet_oct_4_class_retrain_unlearn_CLINCAL_OCT_CLASS_0_LAMDA_0_RETRAIN_14.pth' \
#                  --remain_reg 0.0 \
#                 --run_sota \
#                 --specific_settings \
#                 --gamma .0001 \
#                 --lamda 0


# ###################### OCULOPLASTICS 

# ### CLASS 2

# python main.py --forget_class 2 \
#                --data_name oculoplastic \
#                --model_name resnet \
#                --lr 0.01 \
#                --epoch 30  \
#                --dataset_dir ./data/oculoplastic \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_OCULOPLASTIC_CLASS_2_BEST' \
#                 --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
#                 --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_TRAIN_OCULOPLASTIC_CLASS_0_26.pth' \
#                 --remain_reg 0.0 \
#                 --run_sota \
#                 --specific_settings \
#                 --gamma .0001 \
#                 --lamda 0


## VPF

python main.py --custom_unlearn \
                --oculoplastics \
               --data_name oculoplastic \
               --model_name resnet \
               --lr 0.01 \
               --epoch 30  \
               --dataset_dir ./data/oculoplastic \
               --batch_size 16 \
               --gpu_id 0 \
               --do_unlearning \
               --name  'unlearn_OCULOPLASTIC_CLASS_VPF_BEST' \
                --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
                --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_retrain_OCULOPLASTIC_CLASS_VPF_29.pth' \
                --remain_reg 0.0 \
                --run_sota \
                --specific_settings \
                --gamma .0001 \
                --lamda 0

################ ULTRASOUND

### Class 2 

# python main.py --forget_class 2 \
#                --data_name ultrasound \
#                --model_name resnet \
#                 --lr 0.001 \
#                --epoch 30  \
#                --dataset_dir data/ultrasound_unlearn_oversample/ \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --do_unlearning \
#                --name  'unlearn_ULTRASOUND_CLASS_2_BEST' \
#                 --original_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_original_model_30.pth' \
#                 --retrain_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_retrain_model_class_0_30.pth' \
#                 --remain_reg 0.0 \
#                 --run_sota \
#                 --specific_settings \
#                 --gamma 1 \
#                 --lamda 1