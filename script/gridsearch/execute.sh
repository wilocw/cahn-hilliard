#!/bin/bash
#$ -N CahnHilliardGrid
# Request 16 gigabytes of real memory (RAM) 4 cores *4G = 16
#SBATCH --mem=16G
# Request 4 cores
#SBATCH -c 4
# Email notifications to me@somedomain.com
#SBATCH --mail-user=w.ward@sheffield.ac.uk
# Email notifications if the job fails
#SBATCH --mail-type=all


module load apps/python/conda

# conda create -n pytorch

conda activate pytorch

for SIM_RUN in {1..100}
do
    printf -v RUN_DIR "run_%03d/" $SIM_RUN
    python src/gridsearch/simulate.py -p 100 --landau --out_dir simulations/gridsearch/landau/$RUN_DIR

    python src/gridsearch/simulate.py -p 100 --floryhuggins --out_dir simulations/gridsearch/floryhuggins/$RUN_DIR
done
