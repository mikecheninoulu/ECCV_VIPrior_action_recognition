#!/bin/bash
module purge
module load pytorch/1.1.0
srun --gres=gpu:v100:1,nvme:200 --cpus-per-task=10 --ntasks=1 --time=24:00:00 --partition=gpu --account=project_2001654 --mem-per-cpu=8000 python train.py --sample_duration 16 --cdc_theta 0.7 --model 'tc3d' --model_depth 50 --modality 'RGB_Flow' --batch_size 64 --with_valid 1 --PC 'csc'
