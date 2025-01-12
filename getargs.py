# get training args of other repo

from transformers import TrainingArguments
import torch
import os
import requests
import json
import argparse


parser = argparse.ArgumentParser(description="Parse HF training args")
parser.add_argument('repo', type=str, help='hf repo name')
parser.add_argument('--branch', type=str, default='main', help='branch hash')
parser.add_argument('--ref', type=str, help='repo for compare')
args = parser.parse_args()

def download_file(file_url, file_path):
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception("Failed to download target file from HuggingFace")

dir_path = os.path.dirname(os.path.abspath(__file__))

training_args_url = f'https://huggingface.co/{args.repo}/resolve/{args.branch}/training_args.bin'
training_args_path = os.path.join(dir_path, "training_args.bin")
download_file(training_args_url, training_args_path)
training_args = vars(torch.load(training_args_path))

lora_args_url = f'https://huggingface.co/{args.repo}/raw/{args.branch}/adapter_config.json'
lora_args_path = os.path.join(dir_path, 'lora_args.json')
download_file(lora_args_url, lora_args_path)
with open(lora_args_path, 'r') as f:
    lora_args = json.load(f)

if args.ref:
    ref_training_args_url = f'https://huggingface.co/{args.ref}/resolve/main/training_args.bin'
    ref_training_args_path = os.path.join(dir_path, "ref_training_args.bin")
    download_file(ref_training_args_url, ref_training_args_path)
    ref_training_args = vars(torch.load(ref_training_args_path))
    for key in training_args.keys():
        if training_args[key] != ref_training_args[key]:
            print('\n')
            print('new: ', key, training_args[key])
            print('ref: ', key, ref_training_args[key])


print('----------------------------------------------------')
print(f"target_modules: {lora_args['target_modules']}")
print(f"per_device_train_batch_size: {training_args['per_device_train_batch_size']}")
print(f"gradient_accumulation_steps: {training_args['gradient_accumulation_steps']}")
print(f"num_train_epochs: {training_args['num_train_epochs']}")
print(f"lora_rank: {lora_args['r']}")
print(f"lora_alpha: {lora_args['lora_alpha']}")
print(f"dropout: {lora_args['lora_dropout']}")