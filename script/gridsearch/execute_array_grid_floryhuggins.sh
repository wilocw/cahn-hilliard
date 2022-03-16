#!/bin/bash
#$ -N CahnHilliardGrid
# Request 16 gigabytes of real memory (RAM) 4 cores *4G = 16
#$ -l rmem=64G
# Email notifications to me@somedomain.com
# -M w.ward@sheffield.ac.uk
# -m a
# -t 1-100


module load apps/python/conda
module load libs/cudnn/7.6.5.32/binary-cuda-10.1.243

source activate cahnhilliard

mkdir /data/ci1wow/simulations/gridsearch/floryhuggins/$SGE_TASK_ID

# python src/gridsearch/simulate.py -p 100 --landau --out_dir /data/ci1wow/simulations/gridsearch/landau/$SGE_TASK_ID

python src/simulate.py -p 100 --floryhuggins --out_dir /data/ci1wow/simulations/gridsearch/floryhuggins/$SGE_TASK_ID
