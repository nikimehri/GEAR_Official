

# ##################### Fashion mnist


python main.py --forget_class 0 \
                --data_name fashionmnist \
                --model_name AllCNN \
                --lr 0.01 \
                --epoch 15  \
                --batch_size 64  \
                --gpu_id 0 \
                --sgld \
                --do_unlearning \
                --name "unlearn_fashionmnist_class_0_multiple_gamma_remain_0_lamda_0001_DELETE" \
                --original_model "model_checkpoints/allcnn_fashion/AllCNNfashionmnist_original_model_15.pth" \
                --retrain_model "model_checkpoints/allcnn_fashion/AllCNNfashionmnist_retrain_model_class_0_15.pth" \
                --remain_reg 0.0 \
                --lamda .0001

# python main.py --forget_class 0 \
#                 --data_name fashionmnist \
#                 --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 15  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --do_unlearning \
#                 --name "unlearn_fashionmnist_class_0_multiple_gamma_remain_0_lamda_001" \
#                 --original_model "model_checkpoints/allcnn_fashion/AllCNNfashionmnist_original_model_15.pth" \
#                 --retrain_model "model_checkpoints/allcnn_fashion/AllCNNfashionmnist_retrain_model_class_0_15.pth" \
#                 --remain_reg 0.0 \
#                 --lamda .001

# python main.py --forget_class 0 \
#                 --data_name fashionmnist \
#                 --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 15  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --do_unlearning \
#                 --name "unlearn_fashionmnist_class_0_multiple_gamma_remain_0_lamda_01" \
#                 --original_model "model_checkpoints/allcnn_fashion/AllCNNfashionmnist_original_model_15.pth" \
#                 --retrain_model "model_checkpoints/allcnn_fashion/AllCNNfashionmnist_retrain_model_class_0_15.pth" \
#                 --remain_reg 0.0 \
#                 --lamda .01

# python main.py --forget_class 0 \
#                 --data_name fashionmnist \
#                 --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 15  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --do_unlearning \
#                 --name "unlearn_fashionmnist_class_0_multiple_gamma_remain_0_lamda_1" \
#                 --original_model "model_checkpoints/allcnn_fashion/AllCNNfashionmnist_original_model_15.pth" \
#                 --retrain_model "model_checkpoints/allcnn_fashion/AllCNNfashionmnist_retrain_model_class_0_15.pth" \
#                 --remain_reg 0.0 \
#                 --lamda .1


# python main.py --forget_class 0 \
#                 --data_name fashionmnist \
#                 --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 15  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --do_unlearning \
#                 --name "unlearn_fashionmnist_class_0_multiple_gamma_remain_0_lamda_0" \
#                 --original_model "model_checkpoints/allcnn_fashion/AllCNNfashionmnist_original_model_15.pth" \
#                 --retrain_model "model_checkpoints/allcnn_fashion/AllCNNfashionmnist_retrain_model_class_0_15.pth" \
#                 --remain_reg 0.0 \
#                 --lamda 0



