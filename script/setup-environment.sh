#!/bin/bash
#$ -l rmem=8G
#$ -l gpu=1


module load apps/python/conda
module load libs/cudnn/7.6.5.32/binary-cuda-10.1.243

conda env create -f environment.yml

source activate cahnhilliard

pip install -e ~/git/pdes
pip install -e ~/git/torchfilter

module purge
