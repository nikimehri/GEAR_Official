##### .01

# Chen
python main.py --forget_class 0 \
               --data_name open_source \
               --model_name resnet \
               --lr 0.001 \
               --epoch 15  \
               --dataset_dir ./data/fundus_open_source \
               --batch_size 16 \
               --gpu_id 0 \
               --do_unlearning \
                --name 'CHEN_UNLEARN_FOR_BASELINE_CFP_OS_01' \
               --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
               --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
                --remain_reg 0.0 \
                --run_sota  \
                --percent_to_forget .01 \
                --selective_unlearn \
                --retrain_only


# # RAVI
# python main.py --forget_class 0 \
#                --data_name open_source \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/fundus_open_source \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --do_unlearning \
#                 --name 'RAVI_UNLEARN_FOR_BASELINE_CFP_OS_01' \
#                --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
#                --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --specific_settings \
#                 --lamda .001 \
#                 --gamma .01 \
#                 --percent_to_forget .01  \
#                 --selective_unlearn 



##### .1

# Chen
python main.py --forget_class 0 \
               --data_name open_source \
               --model_name resnet \
               --lr 0.001 \
               --epoch 15  \
               --dataset_dir ./data/fundus_open_source \
               --batch_size 16 \
               --gpu_id 0 \
               --do_unlearning \
                --name 'CHEN_UNLEARN_FOR_BASELINE_CFP_OS_1' \
               --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
               --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
                --remain_reg 0.0 \
                --run_sota   \
                --percent_to_forget .1   \
                --selective_unlearn \
                --retrain_only


# # RAVI
# python main.py --forget_class 0 \
#                --data_name open_source \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/fundus_open_source \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --do_unlearning \
#                 --name 'RAVI_UNLEARN_FOR_BASELINE_CFP_OS_1' \
#                --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
#                --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --specific_settings \
#                 --lamda .001 \
#                 --gamma .01  \
#                 --percent_to_forget .1    \
#                 --selective_unlearn 

##### .25
# Chen
python main.py --forget_class 0 \
               --data_name open_source \
               --model_name resnet \
               --lr 0.001 \
               --epoch 15  \
               --dataset_dir ./data/fundus_open_source \
               --batch_size 16 \
               --gpu_id 0 \
               --do_unlearning \
                --name 'CHEN_UNLEARN_FOR_BASELINE_CFP_OS_25' \
               --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
               --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
                --remain_reg 0.0 \
                --run_sota  \
                --percent_to_forget .25     \
                --selective_unlearn \
                --retrain_only


# # RAVI
# python main.py --forget_class 0 \
#                --data_name open_source \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/fundus_open_source \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --do_unlearning \
#                 --name 'RAVI_UNLEARN_FOR_BASELINE_CFP_OS_25' \
#                --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
#                --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --specific_settings \
#                 --lamda .001 \
#                 --gamma .01 \
#                 --percent_to_forget .25      \
#                 --selective_unlearn




##### .5

# Chen
python main.py --forget_class 0 \
               --data_name open_source \
               --model_name resnet \
               --lr 0.001 \
               --epoch 15  \
               --dataset_dir ./data/fundus_open_source \
               --batch_size 16 \
               --gpu_id 0 \
               --do_unlearning \
                --name 'CHEN_UNLEARN_FOR_BASELINE_CFP_OS_5' \
               --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
               --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
                --remain_reg 0.0 \
                --run_sota  \
                --percent_to_forget .5       \
                --selective_unlearn \
                --retrain_only



# # RAVI
# python main.py --forget_class 0 \
#                --data_name open_source \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/fundus_open_source \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --do_unlearning \
#                 --name 'RAVI_UNLEARN_FOR_BASELINE_CFP_OS_5' \
#                --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
#                --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --specific_settings \
#                 --lamda .001 \
#                 --gamma .01 \
#                 --percent_to_forget .5        \
#                 --selective_unlearn




##### .75

# Chen
python main.py --forget_class 0 \
               --data_name open_source \
               --model_name resnet \
               --lr 0.001 \
               --epoch 15  \
               --dataset_dir ./data/fundus_open_source \
               --batch_size 16 \
               --gpu_id 0 \
               --do_unlearning \
                --name 'CHEN_UNLEARN_FOR_BASELINE_CFP_OS_75' \
               --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
               --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
                --remain_reg 0.0 \
                --run_sota  \
                --percent_to_forget .75          \
                --selective_unlearn \
                --retrain_only



# # RAVI
# python main.py --forget_class 0 \
#                --data_name open_source \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/fundus_open_source \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --do_unlearning \
#                 --name 'RAVI_UNLEARN_FOR_BASELINE_CFP_OS_75' \
#                --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
#                --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --specific_settings \
#                 --lamda .001 \
#                 --gamma .01 \
#                 --percent_to_forget .75         \
#                 --selective_unlearn





##### .001

# Chen
# python main.py --forget_class 0 \
#                --data_name open_source \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/fundus_open_source \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --do_unlearning \
#                 --name 'CHEN_UNLEARN_FOR_BASELINE_CFP_OS_001' \
#                --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
#                --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --run_sota  \
#                 --percent_to_forget .001 \
#                 --selective_unlearn \
#                 --retrain_only


# # # RAVI
# python main.py --forget_class 0 \
#                --data_name open_source \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/fundus_open_source \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --do_unlearning \
#                 --name 'RAVI_UNLEARN_FOR_BASELINE_CFP_OS_001' \
#                --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
#                --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --specific_settings \
#                 --lamda .001 \
#                 --gamma .01 \
#                 --percent_to_forget .001  \
#                 --selective_unlearn 