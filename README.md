```bash
COMMON=(
  --data_name cifar100 --model_name resnet50 --forget_class 0
  --original_model /path/to/original_model.pth
  --retrain_model /path/to/retrain_model.pth
  --batch_size 64 --seed 42 --val_fraction 0.1 --gpu_id 0
)

python main.py "${COMMON[@]}" --do_unlearning --run_sota \
  --feature_contrastive --use_entanglement_weighting --target_layer layer4 \
  --retain_forget_weight 2.0 --forget_forget_weight 3.0 \
  --remain_reg 1.0 --gamma_rep 1.0 --boundary_weight 1.0 \
  --poison_epoch 10 --name gear

python baselines/baseline_main.py "${COMMON[@]}" --method finetune --name finetune

python baselines/baseline_main.py "${COMMON[@]}" --method neggrad --name neggrad

python baselines/baseline_main.py "${COMMON[@]}" --method cfk --name cfk

python baselines/baseline_main.py "${COMMON[@]}" --method euk --name euk

python baselines/baseline_main.py "${COMMON[@]}" --method scrub --name scrub

python baselines/baseline_main.py "${COMMON[@]}" --method delete --name delete

python baselines/baseline_main.py "${COMMON[@]}" --method ssd --name ssd

python baselines/baseline_main.py "${COMMON[@]}" --method coun --name coun

python baselines/baseline_main.py "${COMMON[@]}" --method cu --name cu

python baselines/baseline_main.py "${COMMON[@]}" --method cheng_unlearn --name cheng

python baselines/baseline_main.py "${COMMON[@]}" --method salun --name salun

python baselines/baseline_main.py "${COMMON[@]}" --method bad_teacher --name bad_teacher

python baselines/baseline_main.py "${COMMON[@]}" --method eval_orig --name retrain
```
