#!/bin/bash

qsub -o /dev/null -e /dev/null -M w.ward@sheffield.ac.uk -m ea -b y -l h_rt=00:00:15 -hold_jid $1 -N 'Array_Job_finished' true
