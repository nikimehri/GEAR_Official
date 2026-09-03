##### .01

# Chen
python main.py --forget_class 0 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 25  \
                --batch_size 64  \
                --gpu_id 0 \
                --do_unlearning \
                --name 'CHEN_UNLEARN_FOR_BASELINE_CIFAR_01' \
                --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
                --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
                --remain_reg 0.0 \
                --run_sota  \
                --percent_to_forget .01 \
                --selective_unlearn \
                --retrain_only


# # RAVI
python main.py --forget_class 0 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 25  \
                --batch_size 64  \
                --gpu_id 0 \
                --do_unlearning \
                --name 'RAVI_UNLEARN_FOR_BASELINE_CIFAR_01' \
                --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
                --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
                --remain_reg 0.0 \
                --specific_settings \
                --lamda .001 \
                --gamma .01 \
                --percent_to_forget .01  \
                --selective_unlearn 



##### .1

# Chen
python main.py --forget_class 0 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 25  \
                --batch_size 64  \
                --gpu_id 0 \
                --do_unlearning \
                --name 'CHEN_UNLEARN_FOR_BASELINE_CIFAR_1' \
                --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
                --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
                --remain_reg 0.0 \
                --run_sota   \
                --percent_to_forget .1   \
                --selective_unlearn \
                --retrain_only


# # RAVI
python main.py --forget_class 0 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 25  \
                --batch_size 64  \
                --gpu_id 0 \
                --do_unlearning \
                --name 'RAVI_UNLEARN_FOR_BASELINE_CIFAR_1' \
                --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
                --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
                --remain_reg 0.0 \
                --specific_settings \
                --lamda .001 \
                --gamma .01  \
                --percent_to_forget .1    \
                --selective_unlearn 

##### .25
# Chen
python main.py --forget_class 0 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 25  \
                --batch_size 64  \
                --gpu_id 0 \
                --do_unlearning \
                --name 'CHEN_UNLEARN_FOR_BASELINE_CIFAR_25' \
                --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
                --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
                --remain_reg 0.0 \
                --run_sota  \
                --percent_to_forget .25     \
                --selective_unlearn \
                --retrain_only


# # RAVI
python main.py --forget_class 0 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 25  \
                --batch_size 64  \
                --gpu_id 0 \
                --do_unlearning \
                --name 'RAVI_UNLEARN_FOR_BASELINE_CIFAR_25' \
                --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
                --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
                --remain_reg 0.0 \
                --specific_settings \
                --lamda .001 \
                --gamma .01 \
                --percent_to_forget .25      \
                --selective_unlearn




##### .5

# Chen
python main.py --forget_class 0 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 25  \
                --batch_size 64  \
                --gpu_id 0 \
                --do_unlearning \
                --name 'CHEN_UNLEARN_FOR_BASELINE_CIFAR_5' \
                --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
                --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
                --remain_reg 0.0 \
                --run_sota  \
                --percent_to_forget .5       \
                --selective_unlearn \
                --retrain_only



# # RAVI
python main.py --forget_class 0 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 25  \
                --batch_size 64  \
                --gpu_id 0 \
                --do_unlearning \
                --name 'RAVI_UNLEARN_FOR_BASELINE_CIFAR_5' \
                --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
                --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
                --remain_reg 0.0 \
                --specific_settings \
                --lamda .001 \
                --gamma .01 \
                --percent_to_forget .5        \
                --selective_unlearn




##### .75

# Chen
python main.py --forget_class 0 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 25  \
                --batch_size 64  \
                --gpu_id 0 \
                --do_unlearning \
                --name 'CHEN_UNLEARN_FOR_BASELINE_CIFAR_75' \
                --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
                --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
                --remain_reg 0.0 \
                --run_sota  \
                --percent_to_forget .75          \
                --selective_unlearn \
                --retrain_only



# # RAVI
python main.py --forget_class 0 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 25  \
                --batch_size 64  \
                --gpu_id 0 \
                --do_unlearning \
                --name 'RAVI_UNLEARN_FOR_BASELINE_CIFAR_75' \
                --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
                --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
                --remain_reg 0.0 \
                --specific_settings \
                --lamda .001 \
                --gamma .01 \
                --percent_to_forget .75         \
                --selective_unlearn





##### .001

# Chen
python main.py --forget_class 0 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 25  \
                --batch_size 64  \
                --gpu_id 0 \
                --do_unlearning \
                --name 'CHEN_UNLEARN_FOR_BASELINE_CIFAR_001' \
                --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
                --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
                --remain_reg 0.0 \
                --run_sota  \
                --percent_to_forget .001 \
                --selective_unlearn \
                --retrain_only


# # RAVI
python main.py --forget_class 0 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 25  \
                --batch_size 64  \
                --gpu_id 0 \
                --do_unlearning \
                --name 'RAVI_UNLEARN_FOR_BASELINE_CIFAR_001' \
                --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
                --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
                --remain_reg 0.0 \
                --specific_settings \
                --lamda .001 \
                --gamma .01 \
                --percent_to_forget .001  \
                --selective_unlearn 