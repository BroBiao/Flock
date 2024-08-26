#!/bin/bash

# Training Node Config
read -p "请输入task id：" TASK_ID
read -p "请输入API KEY：" API_KEY
read -p "请输入HuggingFace Token：" HF_TOKEN
read -p "请输入HuggingFace用户名：" HF_USERNAME

# Training Arguments Config
read -p "请输入基础模型名称（默认Qwen/Qwen1.5-0.5B）：" base_model
base_model=${base_model:-"Qwen/Qwen1.5-0.5B"}
read -p "请输入per_device_train_batch_size（默认1）：" per_device_train_batch_size
per_device_train_batch_size=${per_device_train_batch_size:-"1"}
read -p "请输入gradient_accumulation_steps范围（输入值为实际值对2取对数，用空格分隔）：" gradient_accumulation_steps_min gradient_accumulation_steps_max
read -p "请输入num_train_epochs范围（输入值为实际值对2取对数，用空格分隔）：" num_train_epochs_min num_train_epochs_max
read -p "请输入lora_rank范围（输入值为实际值对2取对数，用空格分隔）：" lora_rank_min lora_rank_max
lora_dropouts=(0.1, 0.2, 0.3)
gradient_accumulation_steps_min=$(( 1 << gradient_accumulation_steps_min ))
gradient_accumulation_steps_max=$(( 1 << gradient_accumulation_steps_max ))
num_train_epochs_min=$(( 1 << num_train_epochs_min ))
num_train_epochs_max=$(( 1 << num_train_epochs_max ))
lora_rank_min=$(( 1 << lora_rank_min ))
lora_rank_max=$(( 1 << lora_rank_max ))


cd $HOME/testnet-training-node-quickstart
conda activate training-node
FILE="full_automation.py"
sed -i '
/# generate a random repo id/, $c\
        os.system("rm -rf merged_model")\
        os.system("rm -rf outputs")
' "$FILE"

for (( gradient_accumulation_steps=$gradient_accumulation_steps_min; gradient_accumulation_steps<=$gradient_accumulation_steps_max; gradient_accumulation_steps*=2 )); do
    for (( num_train_epochs=$num_train_epochs_min; num_train_epochs<=$num_train_epochs_max; num_train_epochs*=2 )); do
	for (( lora_rank=$lora_rank_min; lora_rank<=$lora_rank_max; lora_rank*=2 )); do
            lora_alpha_min=$(( $lora_rank >> 2 ))
	    if [ $lora_alpha_min -eq 0 ]; then
		lora_alpha_min=2
	    fi
	    lora_alpha_max=$(( $lora_rank << 1 ))
	    for (( lora_alpha=$lora_alpha_min; lora_alpha<=$lora_alpha_max; lora_alpha*=2 )); do
		for lora_dropout in "${lora_dropouts[@]}"; do
	            cat > training_args.yaml << EOF
$base_model
  per_device_train_batch_size: $per_device_train_batch_size
  gradient_accumulation_steps: $gradient_accumulation_steps
  num_train_epochs: $num_train_epochs
  lora_rank: $lora_rank
  lora_alpha: $lora_alpha
  lora_dropout: $lora_dropout
EOF
cat training_args.yaml >> trainresult.txt
TASK_ID=$TASK_ID FLOCK_API_KEY=$API_KEY HF_TOKEN=$HF_TOKEN CUDA_VISIBLE_DEVICES=0 HF_USERNAME=$HF_USERNAME python full_automation.py | tee localtrain.log
grep 'train_loss' localtrain.log | awk -F', ' '{print $4}' >> trainresult.txt
echo "    "
done
done
done
done
done
