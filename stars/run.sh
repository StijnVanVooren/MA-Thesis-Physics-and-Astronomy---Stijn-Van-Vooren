#!/bin/bash
#SBATCH --job-name=bin_stars
#SBATCH --time=00-71:59:59
#SBATCH --nodes=14
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mail-user=stijn.van.vooren@vub.be
#SBATCH --mail-type=ALL
#SBATCH --output out_bin_stars
#SBATCH --partition=skylake_mpi
#SBATCH --error err_bin_stars

module load Armadillo/11.4.3-foss-2022a

mpic++ -fopenmp -larmadillo -O3 -std=c++20 -o bin_stars binary_stars.cpp 
srun --mpi=pmix_v3 ./bin_stars