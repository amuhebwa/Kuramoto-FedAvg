#!/bin/bash

# Run the Python script with the specified arguments for FedAvg
cd $HOME/Kuramoto-FedAvg/src

#python ./main.py --config_path ./config/fedavg.json --dataset_name mnist --model_name fedavg_mnist
#python ./main.py --config_path ./config/scaffold.json --dataset_name mnist --model_name fedavg_mnist
#python ./main.py --config_path ./config/fedprox.json --dataset_name mnist --model_name fedavg_mnist


#python ./main.py --config_path ./config/fedavg.json --dataset_name cifar10 --model_name vgg11
#python ./main.py --config_path ./config/fedprox.json --dataset_name cifar100 --model_name vgg11
#python ./main.py --config_path ./config/scaffold.json --dataset_name cifar100 --model_name vgg11





python ./main.py --config_path ./config/kuramoto.json --dataset_name cifar10 --model_name vgg11