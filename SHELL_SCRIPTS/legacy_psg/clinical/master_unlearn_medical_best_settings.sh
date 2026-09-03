########################### IODA
#### CFP Class 0

# python main.py --forget_class 0 \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --lr 0.001 \
#                 --epoch 40  \
#                --dataset_dir ./data/fundus_big \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --do_unlearning \
#                --name  'BEST_CFP_CLASS_0_Lam_0_Gam_1e4' \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_40_16_bs_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --lamda 0 \
#                 --gamma .0001 \
#                 --specific_settings \
#                 --run_sota

#### CFP Class Camera



############################## OCT 
#### OCT Class 0





################################## OCULOPLASTICS
#### Oculoplastic Class 0

python main.py --forget_class 0 \
               --data_name oculoplastic \
               --model_name resnet \
               --lr 0.01 \
               --epoch 30  \
               --dataset_dir ./data/oculoplastic \
               --batch_size 16 \
               --gpu_id 0 \
               --do_unlearning \
               --name  'BEST_OCULOPLASTIC_CLASS_0_Lam_1e4_Gam_1' \
                --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
                --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_TRAIN_OCULOPLASTIC_CLASS_0_26.pth' \
                --remain_reg 0.0 \
                --lamda .0001 \
                --gamma 1 \
                --specific_settings 
                


#### Oculoplastic Class 2

# python main.py --forget_class 2 \
#                --data_name oculoplastic \
#                --model_name resnet \
#                --lr 0.01 \
#                --epoch 30  \
#                --dataset_dir ./data/oculoplastic \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --do_unlearning \
#                --name  'BEST_OCULOPLASTIC_CLASS_2_Lam_0_Gam_1e4' \
#                 --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
#                 --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_TRAIN_OCULOPLASTIC_CLASS_0_26.pth' \
#                 --remain_reg 0.0 \
#                 --lamda 0 \
#                 --gamma .0001 \
#                 --specific_settings \
#                --run_sota 



# #### Oculoplastic VPF

# python main.py --custom_unlearn \
#                 --oculoplastics \
#                --data_name oculoplastic \
#                --model_name resnet \
#                --lr 0.01 \
#                --epoch 30  \
#                --dataset_dir ./data/oculoplastic \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --do_unlearning \
#                --name  'BEST_OCULOPLASTIC_CLASS_VPF_Lam_0_Gam_01e4' \
#                 --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
#                 --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_retrain_OCULOPLASTIC_CLASS_VPF_29.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .1 \
#                 --gamma .0001\
#                 --specific_settings \
#                 --run_sota 



#### Ultrasound Class 0



#### Ultrasound Class 2