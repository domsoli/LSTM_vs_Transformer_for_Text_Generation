#!/bin/bash
#
# Set the qstat 'name'
#$ -N Lab_05
#
# Set the current working dir. Generate the output in the current folder.
#$ -cwd
#
# Single output file.
#$ -j y
#
# Enable mail alert
#$ -m ea

# Set the environment
source /nfsd/compneuro/DATASETS/anaconda/etc/profile.d/conda.sh
conda activate

# Commands to be run
python train.py \
--datasetpath="/nfsd/compneuro/DATASETS/shakespeare/shakespeare.txt" \
--crop_len=100 \
--alphabet_len=38 \
--hidden_units=1024 \
--layers_num=3 \
--dropout_prob=0.3 \
--batchsize=154 \
--num_epochs=10 \
--out_dir="pretrained_models/model_10"