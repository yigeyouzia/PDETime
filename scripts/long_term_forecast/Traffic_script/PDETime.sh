#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
model_name=PDETime

# Explicit starting configurations; not a claim of validation-selected paper optima.
# Original source model/loss/sharing/optimizer coverage are preserved.

python -u train_reproduce.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_480_96 \
  --model "$model_name" \
  --data Traffic \
  --features M \
  --protocol paper \
  --seq_len 480 \
  --pred_len 96 \
  --enc_in 862 \
  --c_out 862 \
  --inr_layers 5 \
  --aggregation_layers 1 \
  --patch_length 32 \
  --hidden_features 64 \
  --latent_features 64 \
  --temporal_latent_features 32 \
  --mlp_hidden_layers 3 \
  --mlp_hidden_features 64 \
  --s_cff 1 \
  --ridge_lambda 0.1 \
  --batch_size 32 \
  --train_epochs 10 \
  --learning_rate 1e-4 \
  --seed 2024 \
  --itr 3 \
  --num_workers 0 \
  --device cuda \
  --gpu 0 \
  --selection_metric loss \
  --evaluate_test 1 \
  --checkpoints ./runs/tslib \
  --des source_train_semantics

python -u train_reproduce.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_960_192 \
  --model "$model_name" \
  --data Traffic \
  --features M \
  --protocol paper \
  --seq_len 960 \
  --pred_len 192 \
  --enc_in 862 \
  --c_out 862 \
  --inr_layers 5 \
  --aggregation_layers 1 \
  --patch_length 32 \
  --hidden_features 64 \
  --latent_features 64 \
  --temporal_latent_features 32 \
  --mlp_hidden_layers 3 \
  --mlp_hidden_features 64 \
  --s_cff 1 \
  --ridge_lambda 0.1 \
  --batch_size 32 \
  --train_epochs 10 \
  --learning_rate 1e-4 \
  --seed 2024 \
  --itr 3 \
  --num_workers 0 \
  --device cuda \
  --gpu 0 \
  --selection_metric loss \
  --evaluate_test 1 \
  --checkpoints ./runs/tslib \
  --des source_train_semantics

python -u train_reproduce.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_1680_336 \
  --model "$model_name" \
  --data Traffic \
  --features M \
  --protocol paper \
  --seq_len 1680 \
  --pred_len 336 \
  --enc_in 862 \
  --c_out 862 \
  --inr_layers 5 \
  --aggregation_layers 1 \
  --patch_length 32 \
  --hidden_features 64 \
  --latent_features 64 \
  --temporal_latent_features 32 \
  --mlp_hidden_layers 3 \
  --mlp_hidden_features 64 \
  --s_cff 1 \
  --ridge_lambda 0.1 \
  --batch_size 32 \
  --train_epochs 10 \
  --learning_rate 1e-4 \
  --seed 2024 \
  --itr 3 \
  --num_workers 0 \
  --device cuda \
  --gpu 0 \
  --selection_metric loss \
  --evaluate_test 1 \
  --checkpoints ./runs/tslib \
  --des source_train_semantics

python -u train_reproduce.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_3600_720 \
  --model "$model_name" \
  --data Traffic \
  --features M \
  --protocol paper \
  --seq_len 3600 \
  --pred_len 720 \
  --enc_in 862 \
  --c_out 862 \
  --inr_layers 5 \
  --aggregation_layers 1 \
  --patch_length 32 \
  --hidden_features 64 \
  --latent_features 64 \
  --temporal_latent_features 32 \
  --mlp_hidden_layers 3 \
  --mlp_hidden_features 64 \
  --s_cff 1 \
  --ridge_lambda 0.1 \
  --batch_size 32 \
  --train_epochs 10 \
  --learning_rate 1e-4 \
  --seed 2024 \
  --itr 3 \
  --num_workers 0 \
  --device cuda \
  --gpu 0 \
  --selection_metric loss \
  --evaluate_test 1 \
  --checkpoints ./runs/tslib \
  --des source_train_semantics
