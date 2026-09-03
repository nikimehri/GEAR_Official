

###################### OCULOPLASTICS WHOLE CLASS

# python main.py --forget_class 2 \
#                --data_name oculoplastic \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 30  \
#                --dataset_dir ./data/oculoplastic \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --retrain_only \
#                --name  'TRAIN_OCULOPLASTIC_CLASS_2' \
#                 --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
#                 --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_TRAIN_OCULOPLASTIC_CLASS_2_26.pth' \


# python main.py --forget_class 0 \
#                --data_name oculoplastic \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 30  \
#                --dataset_dir ./data/oculoplastic \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --retrain_only \
#                --name  'TRAIN_OCULOPLASTIC_CLASS_0' \
#                 --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
#                 --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_TRAIN_OCULOPLASTIC_CLASS_0_26.pth' \




### CLASS 2

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
#                --name  'unlearn_OCULOPLASTIC_CLASS_2_LAMDA_0001' \
#                 --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
#                 --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_TRAIN_OCULOPLASTIC_CLASS_2_26.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .0001


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
#                --name  'unlearn_OCULOPLASTIC_CLASS_2_LAMDA_001' \
#                 --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
#                 --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_TRAIN_OCULOPLASTIC_CLASS_2_26.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .001


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
#                --name  'unlearn_OCULOPLASTIC_CLASS_2_LAMDA_01' \
#                 --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
#                 --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_TRAIN_OCULOPLASTIC_CLASS_2_26.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .01


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
#                --name  'unlearn_OCULOPLASTIC_CLASS_2_LAMDA_1' \
#                 --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
#                 --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_TRAIN_OCULOPLASTIC_CLASS_2_26.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .1


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
#                --name  'unlearn_OCULOPLASTIC_CLASS_2_LAMDA_0' \
#                 --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
#                 --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_TRAIN_OCULOPLASTIC_CLASS_2_26.pth' \
#                 --remain_reg 0.0 \
#                 --lamda 0






### CLASS 0

python main.py --forget_class 0 \
               --data_name oculoplastic \
               --model_name resnet \
               --lr 0.01 \
               --epoch 30  \
               --dataset_dir ./data/oculoplastic \
               --batch_size 16 \
               --gpu_id 0 \
               --sgld \
               --do_unlearning \
               --name  'unlearn_OCULOPLASTIC_CLASS_0_LAMDA_0001' \
                --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
                --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_TRAIN_OCULOPLASTIC_CLASS_0_26.pth' \
                --remain_reg 0.0 \
                --lamda .0001


python main.py --forget_class 0 \
               --data_name oculoplastic \
               --model_name resnet \
               --lr 0.01 \
               --epoch 30  \
               --dataset_dir ./data/oculoplastic \
               --batch_size 16 \
               --gpu_id 0 \
               --sgld \
               --do_unlearning \
               --name  'unlearn_OCULOPLASTIC_CLASS_0_LAMDA_001' \
                --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
                --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_TRAIN_OCULOPLASTIC_CLASS_0_26.pth' \
                --remain_reg 0.0 \
                --lamda .001


python main.py --forget_class 0 \
               --data_name oculoplastic \
               --model_name resnet \
               --lr 0.01 \
               --epoch 30  \
               --dataset_dir ./data/oculoplastic \
               --batch_size 16 \
               --gpu_id 0 \
               --sgld \
               --do_unlearning \
               --name  'unlearn_OCULOPLASTIC_CLASS_0_LAMDA_01' \
                --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
                --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_TRAIN_OCULOPLASTIC_CLASS_0_26.pth' \
                --remain_reg 0.0 \
                --lamda .01


python main.py --forget_class 0 \
               --data_name oculoplastic \
               --model_name resnet \
               --lr 0.01 \
               --epoch 30  \
               --dataset_dir ./data/oculoplastic \
               --batch_size 16 \
               --gpu_id 0 \
               --sgld \
               --do_unlearning \
               --name  'unlearn_OCULOPLASTIC_CLASS_0_LAMDA_1' \
                --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
                --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_TRAIN_OCULOPLASTIC_CLASS_0_26.pth' \
                --remain_reg 0.0 \
                --lamda .1


python main.py --forget_class 0 \
               --data_name oculoplastic \
               --model_name resnet \
               --lr 0.01 \
               --epoch 30  \
               --dataset_dir ./data/oculoplastic \
               --batch_size 16 \
               --gpu_id 0 \
               --sgld \
               --do_unlearning \
               --name  'unlearn_OCULOPLASTIC_CLASS_0_LAMDA_0' \
                --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
                --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_TRAIN_OCULOPLASTIC_CLASS_0_26.pth' \
                --remain_reg 0.0 \
                --lamda 0





### CLASS VPF



# python main.py --custom_unlearn \
#                 --oculoplastics \
#                --data_name oculoplastic \
#                --model_name resnet \
#                --lr 0.01 \
#                --epoch 30  \
#                --dataset_dir ./data/oculoplastic \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_OCULOPLASTIC_CLASS_VPF_LAMDA_0001' \
#                 --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
#                 --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_retrain_OCULOPLASTIC_CLASS_VPF_29.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .0001


# python main.py --custom_unlearn \
#                 --oculoplastics \
#                  --data_name oculoplastic \
#                --model_name resnet \
#                --lr 0.01 \
#                --epoch 30  \
#                --dataset_dir ./data/oculoplastic \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_OCULOPLASTIC_CLASS_VPF_LAMDA_001' \
#                 --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
#                 --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_retrain_OCULOPLASTIC_CLASS_VPF_29.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .001


# python main.py --custom_unlearn \
#                 --oculoplastics \
#                --data_name oculoplastic \
#                --model_name resnet \
#                --lr 0.01 \
#                --epoch 30  \
#                --dataset_dir ./data/oculoplastic \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_OCULOPLASTIC_CLASS_VPF_LAMDA_01' \
#                 --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
#                 --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_retrain_OCULOPLASTIC_CLASS_VPF_29.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .01


# python main.py --custom_unlearn \
#                 --oculoplastics \
#                --data_name oculoplastic \
#                --model_name resnet \
#                --lr 0.01 \
#                --epoch 30  \
#                --dataset_dir ./data/oculoplastic \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_OCULOPLASTIC_CLASS_VPF_LAMDA_1' \
#                 --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
#                 --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_retrain_OCULOPLASTIC_CLASS_VPF_29.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .1


# python main.py --custom_unlearn \
#                 --oculoplastics \
#                --data_name oculoplastic \
#                --model_name resnet \
#                --lr 0.01 \
#                --epoch 30  \
#                --dataset_dir ./data/oculoplastic \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_OCULOPLASTIC_CLASS_VPF_LAMDA_0' \
#                 --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
#                 --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_retrain_OCULOPLASTIC_CLASS_VPF_29.pth' \
#                 --remain_reg 0.0 \
#                 --lamda 0

