singularity exec --nv $CMD ~/hyperpoints.img mpirun -n 64 python optimize.py --results_dir results
