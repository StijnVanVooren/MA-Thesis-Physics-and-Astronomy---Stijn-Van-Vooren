#!/bin/bash
#SBATCH --job-name=Local4
#SBATCH --time=00-71:59:59
#SBATCH --nodes=14
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mail-user=stijn.van.vooren@vub.be
#SBATCH --mail-type=ALL
#SBATCH --output out_Local4
#SBATCH --partition=ivybridge_mpi
#SBATCH --error err_Local4

module load Armadillo/11.4.3-foss-2022a

mpic++ -fopenmp -larmadillo -O3 -std=c++20 -o fullSim main.cpp 
srun --mpi=pmix_v3 ./fullSim
