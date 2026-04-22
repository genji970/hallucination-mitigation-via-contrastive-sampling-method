from config.config import parse_args
from train.train import HallucinationTrainer


def main():
    args = parse_args()
    trainer = HallucinationTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()

"""
python main.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --torch_dtype bfloat16 \
  --device_map none \
  --train_dataset_name halueval_qa \
  --eval_dataset_name drop \
  --train_split data \
  --eval_split validation \
  --max_length 8192 \
  --max_new_tokens 16 \
  --max_train_samples -1 \
  --max_eval_samples 128 \
  --num_epochs 1000 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --eval_every 50 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing true \
  --visual_enabled true \
  --report_param_stats true \
  --force_zero_generation_randomness true \
  --full_determinism false \
  --output_root outputs \
  --run_name halueval_llm_match \
  --trainable_target_layer_idx 20 \
  --trainable_ratio 1.0 \
  --trainable_select random \
  --trainable_seed 42 \
  --method_sft_coef 1.0 \
  --method_conf_coef 0.0 \
  --method_prob_margin_target 0.0 \
  --method_bad_ce_target 2.0 \
  --method_halluci_margin_threshold 0.0 \
  --method_gate_scale 1.0 \
  --method_accum_balance true \
  --method_balance_ce_target 1.0 \
  --method_balance_conf_target 1.0 \
  --method_balance_min_scale 0.2 \
  --method_balance_max_scale 5.0 \
  --match_mode llm_semantic \
  --match_judge_max_length 1024 \
  --match_judge_max_new_tokens 4 \
  --match_judge_batch_size 8 \
  --train_use_clean true \
  --train_use_halluci true \
  --train_batch_size 1 \
  --eval_batch_size 1 \
  --sample_trace_enabled true \
  --sample_trace_max_steps 10 \
  --sample_trace_max_samples_per_step 16 \
  --sample_trace_text_max_chars 4000


# new

python main.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --torch_dtype bfloat16 \
  --device_map none \
  --train_dataset_name halueval_qa \
  --eval_dataset_name drop \
  --train_split data \
  --eval_split validation \
  --max_length 8192 \
  --max_new_tokens 16 \
  --max_train_samples -1 \
  --max_eval_samples 128 \
  --num_epochs 1000 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --eval_every 50 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing true \
  --visual_enabled true \
  --report_param_stats true \
  --force_zero_generation_randomness true \
  --full_determinism false \
  --output_root outputs \
  --run_name halueval_llm_match \
  --trainable_target_layer_idx 20 \
  --trainable_ratio 1.0 \
  --trainable_select random \
  --trainable_seed 42 \
  --method_sft_coef 0.0 \
  --method_conf_coef 1.0 \
  --method_prob_margin_target 0.0 \
  --method_bad_ce_target 2.0 \
  --method_halluci_margin_threshold 0.0 \
  --method_gate_scale 1.0 \
  --method_accum_balance true \
  --method_balance_ce_target 1.0 \
  --method_balance_conf_target 1.0 \
  --method_balance_min_scale 0.2 \
  --method_balance_max_scale 5.0 \
  --match_mode llm_semantic \
  --match_judge_max_length 1024 \
  --match_judge_max_new_tokens 4 \
  --match_judge_batch_size 8 \
  --train_use_clean true \
  --train_use_halluci true \
  --train_batch_size 1 \
  --eval_batch_size 1 \
  --sample_trace_enabled true \
  --sample_trace_max_steps 10 \
  --sample_trace_max_samples_per_step 16 \
  --sample_trace_text_max_chars 4000
  

  """