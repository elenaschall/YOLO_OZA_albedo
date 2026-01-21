#!/bin/bash -e
#SBATCH --job-name=YOLO_biodcase_dataset_creation
#SBATCH --account=po_oza.p_oza_ai
#SBATCH --partition=smp
#SBATCH --time=48:00:00
#SBATCH --qos=48h
#SBATCH --ntasks=1
#SBATCH --mail-user=elena.schall@awi.de
#SBATCH --mail-type=ALL     
#SBATCH --output=result_%A_%a.out
#SBATCH --error=result_%A_%a.err

echo "SLURM_JOBID:         $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID:  $SLURM_ARRAY_JOB_ID"

srun python preprocess_data.py
