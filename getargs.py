# get training args of other repo

from transformers import TrainingArguments
import torch
import sys
import os
import requests
import json

if len(sys.argv) == 2:
    hf_repo_name = sys.argv[1]
    branch = 'main'
elif len(sys.argv) == 3:
    hf_repo_name = sys.argv[1]
    branch = sys.argv[2]
else:
    print("Usage: python3 getargs.py hf_repo_name {branch}")
    sys.exit(1)
    

def download_file(file_url, file_path):
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception("Failed to download target file from HuggingFace")

dir_path = os.path.dirname(os.path.abspath(__file__))

training_args_url = f'https://huggingface.co/{hf_repo_name}/resolve/{branch}/training_args.bin'
training_args_path = os.path.join(dir_path, "training_args.bin")
download_file(training_args_url, training_args_path)

lora_args_url = f'https://huggingface.co/{hf_repo_name}/raw/{branch}/adapter_config.json'
lora_args_path = os.path.join(dir_path, 'lora_args.json')
download_file(lora_args_url, lora_args_path)

training_args = torch.load(training_args_path)
if isinstance(training_args, dict):
    training_args = TrainingArguments(**training_args)

with open(lora_args_path, 'r') as f:
    lora_args = json.load(f)

print('----------------------------------------------------')
print(f"per_device_train_batch_size: {training_args.per_device_train_batch_size}")
print(f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
print(f"num_train_epochs: {training_args.num_train_epochs}")
print(f"lora_rank: {lora_args['r']}")
print(f"lora_alpha: {lora_args['lora_alpha']}")
print(f"dropout: {lora_args['lora_dropout']}")

