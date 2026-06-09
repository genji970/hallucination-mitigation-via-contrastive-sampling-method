import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower().strip()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def parse_args():
    p = argparse.ArgumentParser()

    # model
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--model_revision", type=str, default="main")
    p.add_argument("--trust_remote_code", type=str2bool, default=False)
    p.add_argument("--use_fast_tokenizer", type=str2bool, default=True)
    p.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device_map", type=str, default="none")
    p.add_argument("--attn_implementation", type=str, default="")
    p.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    p.add_argument("--full_determinism", type=str2bool, default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf_token", type=str, default="")

    # generation / length
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--method_self_max_new_tokens", type=int, default=0)
    p.add_argument("--do_sample", type=str2bool, default=False)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.0)
    p.add_argument("--force_zero_generation_randomness", type=str2bool, default=True)
    p.add_argument("--prompt_question_max_chars", type=int, default=0)
    p.add_argument("--prompt_reference_max_chars", type=int, default=0)
    p.add_argument("--eval_debug_max_text_chars", type=int, default=0)

    # data
    p.add_argument("--train_dataset_name", type=str, default="halubench")
    p.add_argument("--eval_dataset_name", type=str, default="drop")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default="validation")
    p.add_argument("--eval_dataset_names", type=str, default="")
    p.add_argument("--eval_dataset_exclude_names", type=str, default="")
    p.add_argument("--eval_dataset_splits", type=str, default="")
    p.add_argument("--max_train_samples", type=int, default=-1)
    p.add_argument("--max_eval_samples", type=int, default=128)
    p.add_argument("--train_size", "--train_batch_size", dest="train_size", type=int, default=1)
    p.add_argument("--eval_size", "--eval_batch_size", dest="eval_size", type=int, default=1)

    # training
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--early_stop_step", type=int, default=-1)
    p.add_argument("--output_root", type=str, default="outputs")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--train_pipeline", type=str, default="auto", choices=["auto", "sft_only", "new_method"])
    p.add_argument("--usage_log_every", type=int, default=50)

    # model saving
    p.add_argument("--save_model", type=str2bool, default=True)
    p.add_argument("--save_model_dir", type=str, default="")
    p.add_argument("--save_model_every_eval", type=str2bool, default=False)
    p.add_argument("--push_to_hub", type=str2bool, default=False)
    p.add_argument("--hub_repo_id", type=str, default="")

    # logging
    p.add_argument("--visual_enabled", type=str2bool, default=True)
    p.add_argument("--visual_dir", type=str, default="")
    p.add_argument("--visual_every", type=int, default=1)
    p.add_argument("--report_param_stats", type=str2bool, default=True)
    p.add_argument("--report_param_details", type=str2bool, default=False)
    p.add_argument("--sample_trace_enabled", type=str2bool, default=False)
    p.add_argument("--sample_trace_dir", type=str, default="")
    p.add_argument("--sample_trace_max_steps", type=int, default=-1)
    p.add_argument("--sample_trace_max_samples_per_step", type=int, default=0)
    p.add_argument("--sample_trace_text_max_chars", type=int, default=4000)

    # adaptation
    p.add_argument("--trainable_target_layer_idx", type=int, default=20)
    p.add_argument("--trainable_ratio", type=float, default=0.1)
    p.add_argument("--trainable_select", type=str, default="random", choices=["random", "first"])
    p.add_argument("--trainable_seed", type=int, default=42)

    # formula method
    p.add_argument("--method_sft_coef", type=float, default=1.0)
    p.add_argument("--method_conf_coef", type=float, default=0.0)
    p.add_argument("--method_rank_margin", type=float, default=0.2)
    p.add_argument("--method_topk", type=int, default=8)
    p.add_argument("--method_prob_margin_target", type=float, default=0.0)
    p.add_argument("--method_bad_ce_target", type=float, default=2.0)
    p.add_argument("--method_activation", type=str, default="relu", choices=["softplus", "relu", "sigmoid"])
    p.add_argument("--method_activation_beta", type=float, default=1.0)
    p.add_argument("--method_halluci_margin_threshold", type=float, default=0.0)
    p.add_argument("--method_use_full_seq", type=str2bool, default=False)
    p.add_argument("--method_gate_scale", type=float, default=1.0)
    p.add_argument("--method_accum_balance", type=str2bool, default=False)
    p.add_argument("--method_balance_ce_target", type=float, default=1.0)
    p.add_argument("--method_balance_conf_target", type=float, default=1.0)
    p.add_argument("--method_balance_min_scale", type=float, default=0.1)
    p.add_argument("--method_balance_max_scale", type=float, default=10.0)
    p.add_argument("--train_use_clean", type=str2bool, default=True)
    p.add_argument("--train_use_halluci", type=str2bool, default=True)
    p.add_argument("--train_clean_ce_weight", type=float, default=1.0)
    p.add_argument("--train_halluci_ce_weight", type=float, default=1.0)
    p.add_argument("--match_mode", type=str, default="llm_semantic", choices=["exact", "numeric_or_exact", "llm_semantic"])
    p.add_argument("--match_judge_max_length", type=int, default=1024)
    p.add_argument("--match_judge_max_new_tokens", type=int, default=4)
    p.add_argument("--match_judge_batch_size", type=int, default=8)

    # eval judge
    p.add_argument("--eval_judge_model_name", type=str, default="")
    p.add_argument("--eval_judge_max_length", type=int, default=4096)
    p.add_argument("--eval_judge_max_new_tokens", type=int, default=16)

    args = p.parse_args()

    dm = str(args.device_map).strip().lower()
    args.device_map = None if dm in {"", "none", "null"} else args.device_map
    ai = str(args.attn_implementation).strip().lower()
    args.attn_implementation = None if ai in {"", "none", "null"} else args.attn_implementation

    if args.force_zero_generation_randomness:
        args.do_sample = False
        args.temperature = 0.0
        args.top_p = 0.0

    def _parse_csv_list(text):
        return [item.strip() for item in str(text).split(",") if item.strip()]

    args.eval_dataset_names = _parse_csv_list(args.eval_dataset_names)
    args.eval_dataset_exclude_names = _parse_csv_list(args.eval_dataset_exclude_names)
    args.eval_dataset_splits = _parse_csv_list(args.eval_dataset_splits)

    if args.train_pipeline == "auto":
        args.train_pipeline = "sft_only" if float(args.method_conf_coef) == 0.0 else "new_method"

    if args.train_pipeline == "sft_only":
        args.method_sft_coef = 1.0
        args.method_conf_coef = 0.0
        args.method_accum_balance = False
    elif not args.train_use_clean and not args.train_use_halluci:
        raise ValueError("At least one of --train_use_clean or --train_use_halluci must be true for new_method.")

    mode_subdir = "sft_only" if args.train_pipeline == "sft_only" else "new_method"
    default_run_name = mode_subdir
    run_name = args.run_name.strip() or default_run_name
    args.save_dir = os.path.join(args.output_root, mode_subdir, run_name)
    args.visual_dir = args.visual_dir.strip() or os.path.join(args.save_dir, "visual")
    args.sample_trace_dir = args.sample_trace_dir.strip() or os.path.join(args.save_dir, "sample_trace")

    custom_model_root = args.save_model_dir.strip()
    if custom_model_root:
        args.save_model_dir = os.path.join(custom_model_root, mode_subdir, run_name)
    else:
        args.save_model_dir = os.path.join(os.getcwd(), "saved_models", mode_subdir, run_name)
    args.best_model_dir = os.path.join(args.save_model_dir, "best")
    args.checkpoint_model_dir = os.path.join(args.save_model_dir, "checkpoints")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.visual_dir, exist_ok=True)
    if args.sample_trace_enabled:
        os.makedirs(args.sample_trace_dir, exist_ok=True)
    if args.save_model:
        os.makedirs(args.save_model_dir, exist_ok=True)
    return args
