#!/bin/bash
#SBATCH --time=7-00:00:00                        # Time limit hrs:min:sec
#SBATCH --job-name=liver_ort_det                    # Job name
#SBATCH --qos=a6000_qos
#SBATCH --partition=rtx8000                        # Partition
#SBATCH --nodelist=roentgen                    # Node name
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --output=/projects/disentanglement_methods/reg_model/exp_only_liver_alpha_0.01_ortho_det%j.log   # Standard output and error log
pwd; hostname; date

# Activate conda environment pyenv
source /home/l.estacio/miniconda3/bin/activate pytorch

# Run your command
python /projects/disentanglement_methods/reg_model/main.py