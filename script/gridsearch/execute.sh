#!/usr/bin/env bash

module load apps/python/conda

# conda create -n pytorch

conda activate pytorch

for SIM_RUN in {1..100}
do
    printf -v RUN_DIR "run_%03d/" $SIM_RUN
    python src/gridsearch/simulate.py -p 100 --landau --out_dir simulations/gridsearch/landau/$RUN_DIR

    python src/gridsearch/simulate.py -p 100 --floryhuggins --out_dir simulations/gridsearch/floryhuggins/$RUN_DIR
done
