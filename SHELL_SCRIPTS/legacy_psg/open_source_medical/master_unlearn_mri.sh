## MRI UNLEARN
# python main.py --forget_class 2  \
#                --data_name mri \
#                --model_name resnet \
#                --dataset_dir data/mri_unlearn/ \
#                 --lr 0.001 \
#                 --epoch 15  \
#                 --batch_size 16  \
#                 --retrain_only \
#                 --gpu_id 0 \
#                 --name 'unlearn_MRI_OS_class_3_sota_sgld_closest_points_unlearning_DELETE' \
#                 --original_model 'model_checkpoints/resnet_mri/resnetmri_original_model_15.pth' 

# python main.py --forget_class 0  \
#                --data_name mri \
#                --model_name resnet \
#                --dataset_dir data/mri_unlearn/ \
#                 --lr 0.001 \
#                 --epoch 15  \
#                 --batch_size 16  \
#                 --retrain_only \
#                 --gpu_id 0 \
#                 --name 'unlearn_MRI_OS_class_3_sota_sgld_closest_points_unlearning_DELETE' \
#                 --original_model 'model_checkpoints/resnet_mri/resnetmri_original_model_15.pth' 

# python main.py --forget_class 1  \
#                --data_name mri \
#                --model_name resnet \
#                --dataset_dir data/mri_unlearn/ \
#                 --lr 0.001 \
#                 --epoch 15  \
#                 --batch_size 16  \
#                 --retrain_only \
#                 --gpu_id 0 \
#                 --name 'unlearn_MRI_OS_class_3_sota_sgld_closest_points_unlearning_DELETE' \
#                 --original_model 'model_checkpoints/resnet_mri/resnetmri_original_model_15.pth' 


### MRI UNLEARN
python main.py --forget_class 0 \
               --data_name mri \
               --model_name resnet \
               --dataset_dir data/mri_unlearn/\
                --lr 0.001 \
                --epoch 15  \
                --batch_size 16  \
                --gpu_id 0 \
                --sgld \
                --closest_points \
                --run_sota \
                --do_unlearning \
                --name 'unlearn_MRI_class_0_sota_sgld_closest_points_unlearning_NO_scale' \
                --original_model 'model_checkpoints/resnet_mri/resnetmri_original_model_15.pth' \
                --retrain_model 'model_checkpoints/resnet_mri/resnetmri_retrain_model_class_0_15.pth' 

python main.py --forget_class 1 \
               --data_name mri \
               --model_name resnet \
               --dataset_dir data/mri_unlearn/\
                --lr 0.001 \
                --epoch 15  \
                --batch_size 16  \
                --gpu_id 0 \
                --sgld \
                --closest_points \
                --run_sota \
                --do_unlearning \
                --name 'unlearn_MRI_class_1_sota_sgld_closest_points_unlearning_NO_scale' \
                --original_model 'model_checkpoints/resnet_mri/resnetmri_original_model_15.pth' \
                --retrain_model 'model_checkpoints/resnet_mri/resnetmri_retrain_model_class_1_15.pth' 

python main.py --forget_class 2 \
               --data_name mri \
               --model_name resnet \
               --dataset_dir data/mri_unlearn/\
                --lr 0.001 \
                --epoch 15  \
                --batch_size 16  \
                --gpu_id 0 \
                --sgld \
                --closest_points \
                --run_sota \
                --do_unlearning \
                --name 'unlearn_MRI_class_2_sota_sgld_closest_points_unlearning_NO_scale' \
                --original_model 'model_checkpoints/resnet_mri/resnetmri_original_model_15.pth' \
                --retrain_model 'model_checkpoints/resnet_mri/resnetmri_retrain_model_class_2_15.pth' 


