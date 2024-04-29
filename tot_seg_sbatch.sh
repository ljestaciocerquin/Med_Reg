#!/bin/bash
#SBATCH --time=5-00:00:00                        # Time limit hrs:min:sec
#SBATCH --job-name=tot-seg                     # Job name
#SBATCH --partition=p6000                        # Partition
#SBATCH --nodelist=mariecurie                    # Node name
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --output=/projects/disentanglement_methods/reg_model/tot_seg%j.log   # Standard output and error log
pwd; hostname; date

# Activate conda environment pyenv
source /home/l.estacio/miniconda3/bin/activate tot-seg

# Run your command
python /projects/disentanglement_methods/reg_model/script0_segmentations_of_data.py