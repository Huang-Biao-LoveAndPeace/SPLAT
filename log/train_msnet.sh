#!/bin/bash

python train_msnet.py --data cifar10    --gpu 0 --arch msnet --batch_size 1000 --epochs 310 --lr-type SGDR  --nBlocks 7 --step 4-4-6-2-2-10-2 --nChannels 12 --growthRate 6      --grFactor 1-2-4 --bnFactor 1-2-4 --transFactor 2-5 -j 6

python train_msnet.py --data cifar100    --gpu 0 --arch msnet --batch_size 1000 --epochs 310 --lr-type SGDR  --nBlocks 7 --step 4-4-6-2-2-10-2 --nChannels 12 --growthRate 6      --grFactor 1-2-4 --bnFactor 1-2-4 --transFactor 2-5 -j 6

python train_msnet.py --data tinyimagenet    --gpu 1,2,3 --arch msnet --batch_size 1000 --epochs 310 --lr-type SGDR  --nBlocks 7 --step 4-4-6-2-2-10-2 --nChannels 12 --growthRate 6      --grFactor 1-2-4 --bnFactor 1-2-4 --transFactor 2-5 -j 6

python train_msnet.py --data tinyimagenetsub    --gpu 1,2,3 --arch msnet --batch_size 1000 --epochs 310 --lr-type SGDR  --nBlocks 7 --step 4-4-6-2-2-10-2 --nChannels 12 --growthRate 6      --grFactor 1-2-4 --bnFactor 1-2-4 --transFactor 2-5 -j 6

echo "All commands executed successfully."
