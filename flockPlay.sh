#!/bin/bash

#read -p "请输入task id：" TASK_ID
#read -p "请输入API KEY：" API_KEY
#read -p "请输入HuggingFace Token：" HF_TOKEN
#read -p "请输入HuggingFace用户名：" HF_USERNAME

cd $HOME/testnet-training-node-quickstart

FILE="full_automation.py"
sed -i '
/submit_task/,+3c\
                logger.info(f"REPO_ID={HF_USERNAME}/{hg_repo_id}")
' "$FILE"

conda activate training-node
TASK_ID=$1 FLOCK_API_KEY=$2 HF_TOKEN=$3 CUDA_VISIBLE_DEVICES=0 HF_USERNAME=$4 python full_automation.py
