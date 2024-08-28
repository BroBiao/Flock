#!/bin/bash

# 安装Miniconda
function install_miniconda() {
    if command -v conda &> /dev/null; then
        echo "Miniconda 已安装，conda 命令路径为: $(command -v conda)"
    else
	echo "Miniconda 未安装，开始安装"
        mkdir -p ~/miniconda3
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
        bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
        rm -rf ~/miniconda3/miniconda.sh
        ~/miniconda3/bin/conda init bash
	source ~/miniconda3/bin/activate
    fi
}

# 安装Validator
function install_validator() {
    install_miniconda

    if test -e $HOME/llm-loss-validator; then
        echo "Validator已安装，无需再次安装"
    else
	echo "Validator未安装，开始安装..."
        cd $HOME
        git clone https://github.com/FLock-io/llm-loss-validator.git
        cd llm-loss-validator
        conda create -n llm-loss-validator python==3.10 -y
        conda activate llm-loss-validator
        pip install -r requirements.txt
        echo "安装完成"
    fi
}

# 运行Validator
function run_validator() {
    cd $HOME/llm-loss-validator
    conda activate llm-loss-validator
    cd ./src
    CUDA_VISIBLE_DEVICES=0 bash start.sh --hf_token $1 --flock_api_key $2 --task_id $3 --validation_args_file validation_config.json.example --auto_clean_cache True --lora_only True
}

# Validator守护程序
function start_validator_daemon() {
    read -p "请输入task id：" TASK_ID
    read -p "请输入API KEY：" API_KEY
    read -p "请输入HuggingFace Token：" HF_TOKEN
    while true; do
        run_validator $HF_TOKEN $API_KEY $TASK_ID
    done
}

# 安装Training Node
function install_training_node() {
    install_miniconda

    if test -e $HOME/testnet-training-node-quickstart; then
        echo "Training Node已安装，无需再次安装"
    else
	echo "Training Node未安装，开始安装..."
        cd $HOME
        git clone https://github.com/FLock-io/testnet-training-node-quickstart.git
        cd testnet-training-node-quickstart
        conda create -n training-node python==3.10 -y
        conda activate training-node
        pip install -r requirements.txt
        echo "安装完成"
    fi
}

# 运行Training Node
function run_training_node() {
    cd $HOME/testnet-training-node-quickstart
    conda activate training-node
    TASK_ID=$1 FLOCK_API_KEY=$2 HF_TOKEN=$3 CUDA_VISIBLE_DEVICES=0 HF_USERNAME=$4 python full_automation.py
}

# Training Node守护程序
function start_training_node_daemon() {
    read -p "请输入task id：" TASK_ID
    read -p "请输入API KEY：" API_KEY
    read -p "请输入HuggingFace Token：" HF_TOKEN
    read -p "请输入HuggingFace用户名：" HF_USERNAME
    while true; do
        read -p "请输入基础模型名称（默认Qwen/Qwen1.5-7B）：" base_model
        base_model=${base_model:-"Qwen/Qwen1.5-7B"}
        read -p "请输入per_device_train_batch_size（默认1）：" per_device_train_batch_size
        per_device_train_batch_size=${per_device_train_batch_size:-"1"}
        read -p "请输入gradient_accumulation_steps（默认8）：" gradient_accumulation_steps
        gradient_accumulation_steps=${gradient_accumulation_steps:-"8"}
        read -p "请输入num_train_epochs（默认1）：" num_train_epochs
        num_train_epochs=${num_train_epochs:-"1"}
        read -p "请输入lora_rank（默认4）：" lora_rank
        lora_rank=${lora_rank:-"4"}
        read -p "请输入lora_alpha（默认8）：" lora_alpha
        lora_alpha=${lora_alpha:-"8"}
        read -p "请输入lora_dropout（默认0.1）：" lora_dropout
        lora_dropout=${lora_dropout:-"0.1"}
        cat > $HOME/testnet-training-node-quickstart/training_args.yaml << EOF
$base_model:
  per_device_train_batch_size: $per_device_train_batch_size
  gradient_accumulation_steps: $gradient_accumulation_steps
  num_train_epochs: $num_train_epochs
  lora_rank: $lora_rank
  lora_alpha: $lora_alpha
  lora_dropout: $lora_dropout
EOF
        run_training_node $TASK_ID $API_KEY $HF_TOKEN $HF_USERNAME
        echo "本次训练参数："
        cat $HOME/testnet-training-node-quickstart/training_args.yaml
        echo "请调整训练参数开始下一次训练："
    done
}

# main menu
function main_menu() {
    source ~/miniconda3/bin/activate
    while true; do
	clear;
	echo "请选择要执行的操作："
	echo "1. 安装Validator"
	echo "2. 启动Validator"
	echo "3. 安装Training Node"
	echo "4. 启动Training Node"
	read -p "请输入选项: " OPTION

	case $OPTION in
	1) install_validator ;;
	2) start_validator_daemon ;;
	3) install_training_node ;;
	4) start_training_node_daemon ;;
	*) echo "输入有误！" ;;
	esac
	echo "按任意键返回主菜单..."
	read -n 1
done
}

main_menu
