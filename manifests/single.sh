#!/bin/tcsh
#SBATCH --job-name=vibecheck
#SBATCH -N 1 -n 4
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Load python and venv
cd /sciclone/geograd/stmorse/dtm
module load python/3.12.7
svenv

# Run subtopic clustering
cd src
python -u 5_singlemonth.py \
  --sub_path mbkm_50/bootstrap \
  --year 2020 --month 6 \
  --n_resamples 20 \
  --use_zarr 1 \
  >& ../logs/single_2006.log
