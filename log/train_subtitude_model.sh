#!/bin/bash

{
python train_subtitude_models.py   --dataset cifar10 --batch-size 100 --epochs 200 --lr 0.01 --device cuda:0
} > train_subtitube_cifar10.txt

{
python train_subtitude_models.py   --dataset cifar100 --batch-size 100 --epochs 200 --lr 0.01 --device cuda:0
} > train_subtitube_cifar100.txt

{
python train_subtitude_models.py   --dataset tinyimagenetsub --batch-size 100 --epochs 200 --lr 0.01 --device cuda:0
} > train_subtitube_tinyimagenet.txt


echo "All commands executed successfully."
