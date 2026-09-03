python main.py --forget_class 0 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 25  \
                --batch_size 64  \
                --gpu_id 0 \
                --sgld \
                --do_unlearning \
                --name 'unlearn_cifar_class_0_multiple_gamma_point_0001_lamda_0' \
                --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
                --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
                --remain_reg 0.0001 \
                --lamda 0


# python main.py --forget_class 0 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_0_multiple_gamma_point_001_lamda_0' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
#                 --use_logits \
#                 --remain_reg 0.001 \
#                 --lamda 0


# python main.py --forget_class 0 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_0_multiple_gamma_LOGITS_point_1_lamda_0' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
#                 --use_logits \
#                 --remain_reg 0.1 \
#                 --lamda 0


# python main.py --forget_class 0 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_0_multiple_gamma_preprocess_LOGITS_point_0001_remain' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
#                 --use_logits \
#                 --remain_reg 0.0001 \
#                 --logit_preprocess

# python main.py --forget_class 0 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_0_multiple_gamma_preprocess_LOGITS_point_001_remain' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
#                 --use_logits \
#                 --remain_reg 0.001  \
#                 --logit_preprocess


# python main.py --forget_class 0 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_0_multiple_gamma_preprocess_LOGITS_point_1_remain' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
#                 --use_logits \
#                 --remain_reg 0.1  \
#                 --logit_preprocess




# python main.py --forget_class 0 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_0_multiple_gamma_point_1_remain' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
#                 --remain_reg 0.1

# python main.py --forget_class 0 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_0_multiple_gamma_point_0_remain_BASELINE_COMPARISON' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
#                 --remain_reg 0.0

# python main.py --forget_class 0 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_0_multiple_gamma_LOGITS_point_0_remain' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
#                 --use_logits \
#                 --remain_reg 0.0


# python main.py --forget_class 0 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_0_multiple_gamma_LOGITS_point_5_remain' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
#                 --use_logits \
#                 --remain_reg 0.5


# python main.py --forget_class 0 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_0_multiple_gamma_point_5_remain' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
#                 --remain_reg 0.5
















# python main.py --forget_class 0 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_0_sota_sgld_closest_points_unlearning_NO_scale' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' 



# python main.py --forget_class 1 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_1_sota_sgld_closest_points_unlearning_NO_scale' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_1_25.pth' 


# python main.py --forget_class 2 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_2_sota_sgld_closest_points_unlearning_NO_scale' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_2_25.pth' 

# python main.py --forget_class 3 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_3_sota_sgld_closest_points_unlearning_NO_scale' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_3_25.pth' 


# python main.py --forget_class 4 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_4_sota_sgld_closest_points_unlearning_NO_scale' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_4_25.pth' 

# python main.py --forget_class 5 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_5_sota_sgld_closest_points_unlearning_NO_scale' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_5_25.pth' 

# python main.py --forget_class 6 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_6_sota_sgld_closest_points_unlearning_NO_scale' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_6_25.pth' 

# python main.py --forget_class 7 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_7_sota_sgld_closest_points_unlearning_NO_scale' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_7_25.pth' 

# python main.py --forget_class 8 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_8_sota_sgld_closest_points_unlearning_NO_scale' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_8_25.pth' 

# python main.py --forget_class 9 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                 --name 'unlearn_cifar_class_9_sota_sgld_closest_points_unlearning_NO_scale' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_9_25.pth' 