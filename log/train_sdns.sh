#!/bin/bash

python train_sdns.py --dataset cifar10 --network vgg16bn --vanilla --ic-only

python train_sdns.py --dataset cifar10 --network mobilenet --vanilla --ic-only

python train_sdns.py --dataset cifar10 --network resnet56 --vanilla --ic-only

python train_sdns.py --dataset cifar100 --network vgg16bn --vanilla --ic-only

python train_sdns.py --dataset cifar100 --network mobilenet --vanilla --ic-only

python train_sdns.py --dataset cifar100 --network resnet56 --vanilla --ic-only

python train_sdns.py --dataset tinyimagenet --network vgg16bn --vanilla --ic-only

python train_sdns.py --dataset tinyimagenet --network mobilenet --vanilla --ic-only

python train_sdns.py --dataset tinyimagenet --network resnet56 --vanilla --ic-only

python train_sdns.py --dataset tinyimagenetsub --network vgg16bn --vanilla --ic-only

python train_sdns.py --dataset tinyimagenetsub --network mobilenet --vanilla --ic-only

python train_sdns.py --dataset tinyimagenetsub --network resnet56 --vanilla --ic-only

echo "All commands executed successfully."
