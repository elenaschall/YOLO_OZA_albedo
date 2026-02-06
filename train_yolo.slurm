#!/bin/bash -e
#SBATCH --job-name=YOLO_biodcase_baseline_train_30s
#SBATCH --account=po_oza.p_oza_ai
#SBATCH -p gpu
#SBATCH --gpus=a100:2                 # allocate 2 (out of 4) A100 GPUs; to get 2 (out of 2) A40 GPUs use --gpus=a40:2
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --qos=48h
#SBATCH --mem=0
#SBATCH --ntasks=1
#SBATCH --mail-user=elena.schall@awi.de
#SBATCH --mail-type=ALL     
#SBATCH --output=result_%A_%a.out
#SBATCH --error=result_%A_%a.err 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

## Uncomment the following line to enlarge the stacksize if needed,
##  e.g., if your code crashes with a spurious segmentation fault.
# ulimit -s unlimited

# To be on the safe side, we emphasize that it is pure MPI, no OpenMP threads
export OMP_NUM_THREADS=1

echo "SLURM_JOBID:         $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID:  $SLURM_ARRAY_JOB_ID"
echo CUDA_VISIBILE_DEVICES=$CUDA_VISIBLE_DEVICES

srun python train_yolo.py
