#!/bin/bash

# This script runs the model for a single prediction length (96)
# You can copy and paste this block, changing pred_len to 192, 336, etc.

CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/my_financial_data/ \
  --data_path SPY_with_indicators.csv \
  --model_id SPY_512_96 \
  --model S2IPLLM \
  --data Financial \
  --features M \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 96 \
  --number_variable 5 \
  --n_indicators 6 \
  --target Close \
  --freq d \
  --des 'FinS2IP-LLM_SPY' \
  --itr 1 \
  --d_model 768 \
  --learning_rate 0.0001 \
  --patch_size 16 \
  --stride 8 \
  --add_prompt 1 \
  --prompt_length 4 \
  --batch_size 128 \
  --sim_coef -0.05 \
  --pool_size 1000 \
  --percent 100 \
  --trend_length 96 \
  --seasonal_length 96