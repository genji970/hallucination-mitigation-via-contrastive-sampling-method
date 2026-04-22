#!/usr/bin/env bash
set -euo pipefail

COMMON_ARGS=(
  --model_name Qwen/Qwen2.5-7B-Instruct
  --torch_dtype bfloat16
  --device_map none
  --train_dataset_name halueval_qa
  --train_split data
  --max_length 8192
  --max_new_tokens 16
  --max_train_samples -1
  --max_eval_samples 999999999999999999
  --num_epochs 1
  --max_steps 600
  --learning_rate 1e-5
  --weight_decay 0.0
  --eval_every 100
  --gradient_accumulation_steps 16
  --gradient_checkpointing true
  --visual_enabled true
  --report_param_stats true
  --force_zero_generation_randomness true
  --full_determinism false
  --output_root outputs
  --trainable_ratio 1.0
  --trainable_select random
  --trainable_seed 42
  --method_gate_scale 1.0
  --method_accum_balance true
  --method_balance_ce_target 1.0
  --method_balance_conf_target 1.0
  --method_balance_min_scale 0.2
  --method_balance_max_scale 5.0
  --match_mode llm_semantic
  --match_judge_max_length 65536
  --match_judge_max_new_tokens 64
  --match_judge_batch_size 8
  --train_use_clean true
  --train_use_halluci true
  --train_batch_size 1
  --eval_batch_size 8
  --sample_trace_enabled true
  --sample_trace_max_steps 5
  --sample_trace_max_samples_per_step 16
  --sample_trace_text_max_chars 4000
  --push_to_hub true
  --early_stop_step 100
  --eval_dataset_names faitheval_inconsistent,halueval_dialogue,halueval_summarization,drop,hotpotqa_fullwiki,twowikimultihopqa
  --eval_dataset_splits test,data,data,validation,validation,validation
)

LAYER_IDXS=(14 15 16 17 18 19 20)
BAD_CE_TARGETS=(1.0)
PROB_MARGIN_TARGETS=(0.0)
HALLUCI_THRESHOLDS=(0.0)

for LAYER in "${LAYER_IDXS[@]}"; do
  for BAD_CE in "${BAD_CE_TARGETS[@]}"; do
    for PM_TGT in "${PROB_MARGIN_TARGETS[@]}"; do
      for HM_THR in "${HALLUCI_THRESHOLDS[@]}"; do
        python main.py "${COMMON_ARGS[@]}" \
          --trainable_target_layer_idx "${LAYER}" \
          --hub_repo_id "changgyu/best_trustworthy_model_layer${LAYER}" \
          --run_name "halueval_llm_match_new_1000_layer${LAYER}_badce_${BAD_CE}_pm_${PM_TGT}_hm_${HM_THR}" \
          --train_pipeline new_method \
          --method_sft_coef 0.0 \
          --method_conf_coef 1.0 \
          --method_bad_ce_target "${BAD_CE}" \
          --method_prob_margin_target "${PM_TGT}" \
          --method_halluci_margin_threshold "${HM_THR}"
      done
    done
  done
done