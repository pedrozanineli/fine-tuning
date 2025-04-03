#!/bin/bash
#SBATCH --job-name=mh_ft
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --partition=gpu  
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate mace

for model_size in small medium large; do
    mace_run_train \
        --name="Opt_ZrO2_${model_size}_MACE" \
        --foundation_model="$model_size" \
        --multiheads_finetuning=True \
        --train_file="../data/train.xyz" \
        --valid_fraction=0.05 \
        --test_file="../data/test.xyz" \
        --energy_weight=1.0 \
        --forces_weight=1.0 \
        --E0s="average" \
        --lr=0.01 \
        --scaling="rms_forces_scaling" \
        --batch_size=2 \
        --max_num_epochs=6 \
        --ema \
        --ema_decay=0.99 \
        --amsgrad \
        --default_dtype="float64" \
        --device=cuda \
        --seed=3
	--forces_key="forces" \
	--energy_key="energy" \
	--stress_key="stress"
done

