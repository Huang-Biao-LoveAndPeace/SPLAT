#!/bin/bash

python train_msdnet.py --data cifar10 --arch msdnet --epochs 200 --batch_size 500 --lr 0.1 --gpu 1,2,3

python train_msdnet.py --data cifar100 --arch msdnet --epochs 200 --batch_size 500 --lr 0.1 --gpu 1,2,3

python train_msdnet.py --data tinyimagenet --arch msdnet --epochs 200 --batch_size 500 --lr 0.1 --gpu 1,2,3

python train_msdnet.py --data tinyimagenetsub --arch msdnet --epochs 200 --batch_size 500 --lr 0.1 --gpu 1,2,3

echo "All commands executed successfully."
