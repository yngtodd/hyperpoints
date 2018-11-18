#!/bin/bash

singularity exec --nv $CMD ~/hyperpoints.img mpirun -n 256 python optimize.py --results_dir results
