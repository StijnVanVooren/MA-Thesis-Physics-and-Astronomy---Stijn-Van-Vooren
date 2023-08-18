#!/bin/bash
#PBS -N Stijn
#PBS -l walltime=71:44:00
#PBS -l nodes=14
#PBS -m a

module load Armadillo/11.4.3-foss-2022b


cd $VSC_SCRATCH 

mpic++ -fopenmp -larmadillo -O3 -std=c++20 -o bin_stars gent_binary_stars.cpp 
srun --mpi=pmix_v3 ./bin_stars