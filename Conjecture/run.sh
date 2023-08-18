#!/bin/bash
#SBATCH --job-name=conjecture1
#SBATCH --time=00-71:59:59
#SBATCH --nodes=22
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mail-user=stijn.van.vooren@vub.be
#SBATCH --mail-type=ALL
#SBATCH --output out_conjecture1
#SBATCH --partition=broadwell
#SBATCH --error err_conjecture1

module load Armadillo/11.4.3-foss-2022a

mpic++ -fopenmp -larmadillo -O3 -std=c++20 -o conjecture1 scaleFree.cpp 
srun --mpi=pmix_v3 ./conjecture1
