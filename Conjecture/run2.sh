#!/bin/bash
#SBATCH --job-name=conjectures_randomTree
#SBATCH --time=00-71:59:59
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mail-user=stijn.van.vooren@vub.be
#SBATCH --mail-type=ALL
#SBATCH --output out_conjectures_randomTree
#SBATCH --partition=ivybridge_mpi
#SBATCH --error err_conjectures_randomTree

module load Armadillo/11.4.3-foss-2022a
module load igraph/0.10.3-foss-2022a

mpic++ -fopenmp -larmadillo -ligraph -O3 -std=c++20 -o conjectures_randomTree conjectures.cpp
srun --mpi=pmix_v3 ./conjectures_randomTree