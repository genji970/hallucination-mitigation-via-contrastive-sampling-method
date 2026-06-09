import os
import shutil
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from util.util import compute_grad_stats, format_train_step_debug, limit_text, safe_filename
from util.visual import VisualLogger, append_jsonl, build_param_report, format_param_report, ensure_dir, save_json


class HallucinationTrainer:
    def __init__(self, args):
        self.args = args
        self.validate_args()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.base_model = None
        self.base_tokenizer = None
        self.eval_judge_model = None
        self.eval_judge_tokenizer = None
        self.objective = None
        self.train_dataset = None
        self.eval_dataset = None
        self.eval_job_rows = []
        self.train_loader = None
        self.optimizer = None
        self.global_step = 0
        self.micro_step = 0
        self.visual = None
        self.best_checkpoint_path = os.path.join(self.args.save_dir, "best_checkpoint.json")
        self.best_comparison = None
        self.usage_summary_path = os.path.join(self.args.save_dir, "usage_summary_50step.jsonl")
        self.usage_running_path = os.path.join(self.args.save_dir, "usage_running_summary.json")
        self.running_total_seen = 0
        self.running_union_used = 0
        self.running_conf_used = 0
        self.running_base_mismatch = 0
        self.interval_total_seen = 0
        self.interval_union_used = 0
        self.interval_conf_used = 0
        self.interval_base_mismatch = 0
        self.interval_step_count = 0
        self.build_all()

    def validate_args(self):
        if self.args.device_map is not None:
            raise ValueError("This training code is single-device only. Use --device_map none.")
        if self.args.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")

    def build_all(self):
        self.build_visual_logger()
        self.load_models()
        self.load_objective()
        self.load_datasets()
        self.build_dataloader()
        self.build_optimizer()
        self.report_parameter_stats()

    def build_visual_logger(self):
        if not getattr(self.args, "visual_enabled", True):
            self.visual = None
            return
        self.visual = VisualLogger(self.args.visual_dir)
        self.visual.log_args(self.args)
        print(f"visual_dir={self.args.visual_dir}")
        if getattr(self.args, "sample_trace_enabled", False):
            print(f"sample_trace_dir={self.args.sample_trace_dir}")
        if getattr(self.args, "train_pipeline", "new_method") == "new_method":
            print(f"usage_summary_path={self.usage_summary_path}")

    def freeze_model(self, model):
        model.eval()
        model.config.use_cache = True
        for p in model.parameters():
            p.requires_grad = False

    def load_models(self):
        from model.model_load import load_model_and_tokenizer

        self.model, self.tokenizer = load_model_and_tokenizer(self.args)
        self.model.to(self.device)
        if getattr(self.args, "gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

        self.base_model, self.base_tokenizer = load_model_and_tokenizer(self.args)
        self.freeze_model(self.base_model)
        self.base_model.to("cpu")

        judge_model_name = getattr(self.args, "eval_judge_model_name", "") or self.args.model_name
        if judge_model_name == self.args.model_name:
            self.eval_judge_model = self.base_model
            self.eval_judge_tokenizer = self.base_tokenizer
        else:
            self.eval_judge_model, self.eval_judge_tokenizer = load_model_and_tokenizer(
                self.args,
                model_name=judge_model_name,
            )
            self.freeze_model(self.eval_judge_model)
            self.eval_judge_model.to("cpu")
        print(f"eval_judge_model_name={judge_model_name}")

    def load_objective(self):
        from algorithm.algorithm import FormulaHallucinationObjective

        self.objective = FormulaHallucinationObjective(
            self.args,
            self.model,
            self.tokenizer,
            base_model=self.base_model,
            base_tokenizer=self.base_tokenizer,
        )
        self.objective.to(self.device)
        print(self.objective.describe())

    def report_parameter_stats(self):
        report = build_param_report(
            self.model,
            objective_module=self.objective,
            include_details=getattr(self.args, "report_param_details", False),
        )
        if self.visual is not None:
            self.visual.log_param_report(report)
        if getattr(self.args, "report_param_stats", True):
            print(format_param_report(report, show_details=getattr(self.args, "report_param_details", False)))

    def save_model_snapshot(self, tag: str = "final", save_dir: str | None = None, extra_info: Dict[str, Any] | None = None):
        if not getattr(self.args, "save_model", True):
            return None
        step_tag = f"step_{int(self.global_step):06d}"
        safe_tag = str(tag).strip() or "final"
        if save_dir is None:
            if safe_tag == "best":
                save_dir = self.args.best_model_dir
            elif safe_tag == "final":
                save_dir = os.path.join(self.args.save_model_dir, "final")
            else:
                save_dir = os.path.join(self.args.checkpoint_model_dir, f"{safe_tag}__{step_tag}")
        ensure_dir(save_dir)
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            if hasattr(self.model.generation_config, "top_k"):
                self.model.generation_config.top_k = None
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        info = {
            "tag": safe_tag,
            "step": int(self.global_step),
            "train_pipeline": str(getattr(self.args, "train_pipeline", "new_method")),
            "model_name": str(getattr(self.model, "name_or_path", getattr(self.model.config, "_name_or_path", self.args.model_name))),
            "source_save_dir": self.args.save_dir,
            "model_save_root": self.args.save_model_dir,
            "model_save_dir": save_dir,
        }
        if extra_info:
            info.update(extra_info)
        save_json(os.path.join(save_dir, "save_info.json"), info)
        print(f"model_saved_dir={save_dir}")
        return save_dir

    def _push_to_hub(self, step: int):
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            repo_id = self.args.hub_repo_id
            revision = f"step-{step}"
            api.create_repo(repo_id=repo_id, exist_ok=True, private=True)
            try:
                api.create_branch(repo_id=repo_id, branch=revision)
            except Exception:
                pass
            save_dir = self.args.best_model_dir if os.path.exists(self.args.best_model_dir) else os.path.join(self.args.save_model_dir, "final")
            if not os.path.exists(save_dir):
                tmp_dir = os.path.join(self.args.save_model_dir, f"_hub_tmp_step_{step}")
                self.save_model_snapshot(tag="hub_push", save_dir=tmp_dir, extra_info={"push_step": step})
                save_dir = tmp_dir
            api.upload_folder(
                folder_path=save_dir,
                repo_id=repo_id,
                revision=revision,
                commit_message=f"step {step}",
            )
            print(f"[hub] pushed to {repo_id} branch={revision}")
        except Exception as e:
            print(f"[hub] push failed: {e}")

    def _comparison_rank_tuple(self, comparison: Dict[str, Any]):
        adapted = comparison.get("adapted", {})
        label_diff = comparison.get("label_diff", {})
        return (
            int(adapted.get("num_correct", 0)),
            -int(adapted.get("num_hallucinated", 0)),
            int(label_diff.get("improved", 0)),
            -int(label_diff.get("worsened", 0)),
        )

    def _is_better_comparison(self, comparison: Dict[str, Any]) -> bool:
        if self.best_comparison is None:
            return True
        return self._comparison_rank_tuple(comparison) > self._comparison_rank_tuple(self.best_comparison)

    def _copy_eval_artifacts_to_best(self, eval_dir: str):
        ensure_dir(self.args.best_model_dir)
        artifact_names = [
            "comparison.json",
            "base_predictions.jsonl",
            "adapted_predictions.jsonl",
            "base_judge.jsonl",
            "adapted_judge.jsonl",
        ]
        for name in artifact_names:
            src = os.path.join(eval_dir, name)
            dst = os.path.join(self.args.best_model_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, dst)

    def _update_best_checkpoint(self, comparison: Dict[str, Any], eval_dir: str):
        if not self._is_better_comparison(comparison):
            return False
        self.best_comparison = comparison
        adapted = comparison.get("adapted", {})
        label_diff = comparison.get("label_diff", {})
        best_record = {
            "step": int(comparison.get("step", self.global_step)),
            "train_pipeline": str(getattr(self.args, "train_pipeline", "new_method")),
            "eval_dir": eval_dir,
            "model_dir": self.args.best_model_dir,
            "score": {
                "num_correct": int(adapted.get("num_correct", 0)),
                "num_hallucinated": int(adapted.get("num_hallucinated", 0)),
                "hallucination_rate": float(adapted.get("hallucination_rate", 0.0)),
                "improved": int(label_diff.get("improved", 0)),
                "worsened": int(label_diff.get("worsened", 0)),
            },
            "comparison": comparison,
        }
        save_json(self.best_checkpoint_path, best_record)
        self._copy_eval_artifacts_to_best(eval_dir)
        self.save_model_snapshot(
            tag="best",
            save_dir=self.args.best_model_dir,
            extra_info={
                "best_step": int(comparison.get("step", self.global_step)),
                "best_num_correct": int(adapted.get("num_correct", 0)),
                "best_num_hallucinated": int(adapted.get("num_hallucinated", 0)),
                "best_hallucination_rate": float(adapted.get("hallucination_rate", 0.0)),
                "best_improved": int(label_diff.get("improved", 0)),
                "best_worsened": int(label_diff.get("worsened", 0)),
            },
        )
        print(
            f"[best] step={best_record['step']} "
            f"correct={best_record['score']['num_correct']} "
            f"hallucinated={best_record['score']['num_hallucinated']} "
            f"improved={best_record['score']['improved']} "
            f"worsened={best_record['score']['worsened']}"
        )
        return True


    def _parse_csv_arg(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            out = []
            for item in value:
                out.extend(self._parse_csv_arg(item))
            return out
        return [part.strip() for part in str(value).split(",") if part.strip()]

    def _available_loader_dataset_names(self, loader) -> List[str]:
        names = []
        for attr in dir(loader):
            if not attr.startswith("load_") or attr == "load_by_name":
                continue
            fn = getattr(loader, attr, None)
            if callable(fn):
                names.append(attr[len("load_"):])
        return sorted(set(names))

    def _choose_eval_split(self, dataset_name: str, ds) -> str:
        if not hasattr(ds, "keys"):
            return getattr(self.args, "eval_split", "validation")
        splits = list(ds.keys())
        preferred = {
            "drop": ["validation", "test", "data", "train"],
            "hotpotqa_fullwiki": ["validation", "test", "train"],
            "twowikimultihopqa": ["validation", "test", "train"],
            "faitheval_inconsistent": ["test", "validation", "train"],
            "halueval_qa": ["data", "validation", "test", "train"],
            "halueval_dialogue": ["data", "validation", "test", "train"],
            "halueval_summarization": ["data", "validation", "test", "train"],
            "truthfulqa": ["train"],
            "ragtruth": ["train"],
            "anah": ["train"],
        }.get(dataset_name, [])
        for cand in preferred + ["validation", "test", "data", "dev", "train"]:
            if cand in splits:
                return cand
        return splits[0]

    def _build_eval_dataset_names(self, loader) -> List[str]:
        include = self._parse_csv_arg(getattr(self.args, "eval_dataset_names", ""))
        exclude = set(self._parse_csv_arg(getattr(self.args, "eval_dataset_exclude_names", "")))
        train_name = str(getattr(self.args, "train_dataset_name", "")).strip()
        if include:
            names = include
        else:
            names = self._available_loader_dataset_names(loader)
        out = []
        for name in names:
            if name == train_name:
                continue
            if name in exclude:
                continue
            out.append(name)
        return out

    def _aggregate_eval_comparisons(self, comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not comparisons:
            return {
                "step": int(self.global_step),
                "num_jobs": 0,
                "base": {"num_samples": 0, "num_hallucinated": 0, "hallucination_rate": 0.0, "num_correct": 0, "num_failures": 0},
                "adapted": {"num_samples": 0, "num_hallucinated": 0, "hallucination_rate": 0.0, "num_correct": 0, "num_failures": 0},
                "delta": {"hallucination_rate": 0.0, "num_hallucinated": 0, "num_correct": 0},
                "output_diff": {"same": 0, "different": 0},
                "label_diff": {"improved": 0, "worsened": 0},
                "eval_time_sec": 0.0,
                "datasets": [],
            }

        def sum_nested(section: str, key: str) -> int:
            return sum(int(comp.get(section, {}).get(key, 0)) for comp in comparisons)

        base_num_samples = sum_nested("base", "num_samples")
        base_num_hall = sum_nested("base", "num_hallucinated")
        base_num_fail = sum_nested("base", "num_failures")
        adapted_num_samples = sum_nested("adapted", "num_samples")
        adapted_num_hall = sum_nested("adapted", "num_hallucinated")
        adapted_num_fail = sum_nested("adapted", "num_failures")

        base = {
            "num_samples": int(base_num_samples),
            "num_hallucinated": int(base_num_hall),
            "hallucination_rate": 0.0 if base_num_samples == 0 else float(base_num_hall) / float(base_num_samples),
            "num_correct": int(base_num_samples - base_num_hall),
            "num_failures": int(base_num_fail),
        }
        adapted = {
            "num_samples": int(adapted_num_samples),
            "num_hallucinated": int(adapted_num_hall),
            "hallucination_rate": 0.0 if adapted_num_samples == 0 else float(adapted_num_hall) / float(adapted_num_samples),
            "num_correct": int(adapted_num_samples - adapted_num_hall),
            "num_failures": int(adapted_num_fail),
        }
        return {
            "step": int(self.global_step),
            "num_jobs": int(len(comparisons)),
            "base": base,
            "adapted": adapted,
            "delta": {
                "hallucination_rate": round(adapted["hallucination_rate"] - base["hallucination_rate"], 6),
                "num_hallucinated": int(adapted["num_hallucinated"] - base["num_hallucinated"]),
                "num_correct": int(adapted["num_correct"] - base["num_correct"]),
            },
            "output_diff": {
                "same": int(sum(int(comp.get("output_diff", {}).get("same", 0)) for comp in comparisons)),
                "different": int(sum(int(comp.get("output_diff", {}).get("different", 0)) for comp in comparisons)),
            },
            "label_diff": {
                "improved": int(sum(int(comp.get("label_diff", {}).get("improved", 0)) for comp in comparisons)),
                "worsened": int(sum(int(comp.get("label_diff", {}).get("worsened", 0)) for comp in comparisons)),
            },
            "eval_time_sec": float(sum(float(comp.get("eval_time_sec", 0.0)) for comp in comparisons)),
            "datasets": [
                {
                    "dataset_name": comp.get("dataset_name", ""),
                    "split_name": comp.get("split_name", ""),
                    "base": comp.get("base", {}),
                    "adapted": comp.get("adapted", {}),
                    "delta": comp.get("delta", {}),
                    "label_diff": comp.get("label_diff", {}),
                }
                for comp in comparisons
            ],
        }

    def pick_split(self, ds, split_name):
        if hasattr(ds, "keys"):
            return ds[split_name] if split_name in ds else ds[list(ds.keys())[0]]
        return ds

    def filter_answered(self, ds):
        keep = []
        for i in range(len(ds)):
            answer = "" if ds[i]["answer"] is None else str(ds[i]["answer"]).strip()
            if answer != "":
                keep.append(i)
        return ds.select(keep)

    def maybe_slice(self, ds, max_samples):
        if max_samples is None or max_samples < 0 or len(ds) <= max_samples:
            return ds
        return ds.select(range(max_samples))

    def dataset_overlap_report(self, train_rows: List[Dict[str, str]], eval_rows: List[Dict[str, str]]):
        train_keys = {
            (
                str(r.get("question", "")).strip(),
                str(r.get("reference", "")).strip(),
                str(r.get("answer", "")).strip(),
            )
            for r in train_rows
        }
        eval_keys = {
            (
                str(r.get("question", "")).strip(),
                str(r.get("reference", "")).strip(),
                str(r.get("answer", "")).strip(),
            )
            for r in eval_rows
        }
        overlap = train_keys & eval_keys
        print(f"dataset_overlap_exact={len(overlap)}")


    def load_datasets(self):
        from data_load.data_load import HallucinationDatasetLoader

        loader = HallucinationDatasetLoader()
        train_raw = loader.load_by_name(self.args.train_dataset_name)

        self.train_dataset = self.pick_split(train_raw, self.args.train_split)
        self.train_dataset = self.filter_answered(self.train_dataset)
        self.train_dataset = self.maybe_slice(self.train_dataset, self.args.max_train_samples)

        eval_dataset_names = self._build_eval_dataset_names(loader)
        explicit_splits = list(getattr(self.args, "eval_dataset_splits", []) or [])
        self.eval_job_rows = []
        all_eval_rows: List[Dict[str, str]] = []
        for idx, dataset_name in enumerate(eval_dataset_names):
            ds_raw = loader.load_by_name(dataset_name)
            if idx < len(explicit_splits) and explicit_splits[idx]:
                split_name = explicit_splits[idx]
            else:
                split_name = self._choose_eval_split(dataset_name, ds_raw)
            ds = self.pick_split(ds_raw, split_name)
            ds = self.filter_answered(ds)
            ds = self.maybe_slice(ds, self.args.max_eval_samples)
            rows = [ds[i] for i in range(len(ds))]
            self.eval_job_rows.append({
                "dataset_name": dataset_name,
                "split_name": split_name,
                "rows": rows,
                "num_rows_total": int(len(ds)),
            })
            all_eval_rows.extend(rows)

        self.eval_dataset = all_eval_rows

        print(f"train_dataset_name={self.args.train_dataset_name}")
        print(f"train_samples={len(self.train_dataset)}")
        print(f"eval_jobs={len(self.eval_job_rows)}")
        for info in self.eval_job_rows:
            print(
                f"eval_dataset_name={info['dataset_name']} split={info['split_name']} "
                f"eval_samples={len(info['rows'])}"
            )
        self.dataset_overlap_report(
            [self.train_dataset[i] for i in range(len(self.train_dataset))],
            all_eval_rows,
        )

    def build_dataloader(self):

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_size,
            shuffle=True,
            collate_fn=self.objective.collate,
        )

    def build_optimizer(self):
        trainable_params = [p for p in self.objective.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found.")
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.trainable_params = trainable_params
        print(f"optimizer_trainable_param_count={sum(p.numel() for p in trainable_params)}")

    def move_base_model_to_device(self):
        self.base_model.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def offload_base_model_to_cpu(self):
        self.base_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def precompute_base_answers(self, window_batches) -> List[Tuple[List[str], List[List[int]]]]:
        cached = []
        self.move_base_model_to_device()
        try:
            for batch in window_batches:
                answers, token_ids = self.objective.generate_base_answers(batch)
                cached.append((answers, token_ids))
        finally:
            self.offload_base_model_to_cpu()
        return cached

    def compute_window_balance(self, window_stats: List[Dict[str, float]]):
        scales = {"ce": 1.0, "conf": 1.0}
        if not getattr(self.args, "method_accum_balance", False):
            return scales

        def mean_from_stats(num_key: str, den_key: str):
            num = sum(float(s.get(num_key, 0.0)) for s in window_stats)
            den = sum(float(s.get(den_key, 0.0)) for s in window_stats)
            if den <= 0.0:
                return None
            return num / max(den, 1e-8)

        raw_ce = mean_from_stats("ce_num", "ce_denom")
        raw_conf = mean_from_stats("conf_num", "conf_denom")

        observed = {
            "ce": None if raw_ce is None else float(self.args.method_sft_coef) * raw_ce,
            "conf": None if raw_conf is None else float(self.args.method_conf_coef) * raw_conf,
        }
        targets = {
            "ce": max(float(getattr(self.args, "method_balance_ce_target", 1.0)), 0.0),
            "conf": max(float(getattr(self.args, "method_balance_conf_target", 1.0)), 0.0),
        }
        active = [k for k in ("ce", "conf") if observed[k] is not None and observed[k] > 1e-8 and targets[k] > 0.0]
        if not active:
            return scales

        total_obs = sum(observed[k] for k in active)
        total_tgt = sum(targets[k] for k in active)
        min_scale = float(getattr(self.args, "method_balance_min_scale", 0.1))
        max_scale = float(getattr(self.args, "method_balance_max_scale", 10.0))
        for k in active:
            desired = total_obs * targets[k] / max(total_tgt, 1e-8)
            scales[k] = max(min(desired / max(observed[k], 1e-8), max_scale), min_scale)
        return scales

    def build_window_normalizers(self, window_stats: List[Dict[str, float]]):
        return {
            "ce": max(sum(float(s.get("ce_denom", 0.0)) for s in window_stats), 0.0),
            "conf": max(sum(float(s.get("conf_denom", 0.0)) for s in window_stats), 0.0),
        }

    def summarize_window(self, window_stats: List[Dict[str, float]], balance_scales: Dict[str, float]):
        device = next(self.model.parameters()).device

        def sum_key(key: str):
            return sum(float(s.get(key, 0.0)) for s in window_stats)

        def mean_from_num_den(num_key: str, den_key: str):
            num = sum_key(num_key)
            den = sum_key(den_key)
            return 0.0 if den <= 0.0 else num / max(den, 1e-8)

        total_batch = int(sum(int(s.get("batch_size", 0)) for s in window_stats))
        ce_base_loss = mean_from_num_den("ce_num", "ce_denom")
        conf_loss = mean_from_num_den("conf_num", "conf_denom")

        loss = (
            float(self.args.method_sft_coef) * float(balance_scales["ce"]) * ce_base_loss
            + float(self.args.method_conf_coef) * float(balance_scales["conf"]) * conf_loss
        )

        def weighted_mean(key: str):
            if total_batch <= 0:
                return 0.0
            return sum(float(s.get(key, 0.0)) * float(s.get("batch_size", 0)) for s in window_stats) / max(total_batch, 1)

        used_ce = int(sum_key("num_used_ce"))
        used_conf = int(sum_key("num_used_conf"))
        used_union = int(sum_key("num_used_union"))
        base_mismatch = int(sum_key("num_base_mismatch"))
        return {
            "loss": torch.tensor(loss, device=device),
            "ce_base_loss": torch.tensor(ce_base_loss, device=device),
            "conf_loss": torch.tensor(conf_loss, device=device),
            "ce_contrib": torch.tensor(float(self.args.method_sft_coef) * float(balance_scales["ce"]) * ce_base_loss, device=device),
            "conf_contrib": torch.tensor(float(self.args.method_conf_coef) * float(balance_scales["conf"]) * conf_loss, device=device),
            "raw_mag_mean": torch.tensor(weighted_mean("raw_mag_mean"), device=device),
            "mag_mean": torch.tensor(weighted_mean("mag_mean"), device=device),
            "mag_pos_mean": torch.tensor(weighted_mean("mag_pos_mean"), device=device),
            "mag_neg_mean": torch.tensor(weighted_mean("mag_neg_mean"), device=device),
            "rank_loss_mean": torch.tensor(weighted_mean("rank_loss_mean"), device=device),
            "scale_ce": float(balance_scales["ce"]),
            "scale_conf": float(balance_scales["conf"]),
            "num_halluci": int(sum_key("num_halluci")),
            "num_clean": int(sum_key("num_clean")),
            "num_selected": int(sum_key("num_selected")),
            "num_used_ce": used_ce,
            "num_used_conf": used_conf,
            "num_used_union": used_union,
            "num_conf_eligible": int(sum_key("num_conf_eligible")),
            "num_base_mismatch": base_mismatch,
            "num_total_samples": total_batch,
            "batch_size": total_batch,
        }

    def move_eval_models_to_device(self):
        self.base_model.to(self.device)
        if self.eval_judge_model is not None and self.eval_judge_model is not self.base_model:
            self.eval_judge_model.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def offload_eval_models_to_cpu(self):
        self.base_model.to("cpu")
        if self.eval_judge_model is not None and self.eval_judge_model is not self.base_model:
            self.eval_judge_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _trace_enabled_for_step(self, next_step: int) -> bool:
        if not getattr(self.args, "sample_trace_enabled", False):
            return False
        max_steps = int(getattr(self.args, "sample_trace_max_steps", -1))
        return max_steps < 0 or next_step <= max_steps

    def _clip_trace_text(self, text: Any) -> str:
        return limit_text(text, int(getattr(self.args, "sample_trace_text_max_chars", 0)))

    def _window_summary_to_python(self, window_summary: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in window_summary.items():
            if torch.is_tensor(v):
                out[k] = float(v.detach().float().item())
            else:
                out[k] = v
        return out

    def _build_sample_trace_rows(
        self,
        *,
        epoch: int,
        next_step: int,
        batch_index: int,
        batch: Dict[str, Any],
        base_answers: List[str],
        base_answer_token_ids: List[List[int]],
        loss_dict: Dict[str, Any],
        ce_term_value: float,
        conf_term_value: float,
        batch_loss_value: float,
        balance_scales: Dict[str, float],
        window_norm: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        rows = []
        sample_metrics = loss_dict.get("sample_metrics", [])
        if not sample_metrics:
            return rows
        for j, metric in enumerate(sample_metrics):
            sample_ce_num = float(metric.get("sample_ce_num", 0.0))
            sample_conf_num = float(metric.get("sample_conf_num", 0.0))
            eff_ce = 0.0
            eff_conf = 0.0
            if window_norm["ce"] > 0.0 and float(self.args.method_sft_coef) != 0.0:
                eff_ce = float(self.args.method_sft_coef) * float(balance_scales["ce"]) * sample_ce_num / max(window_norm["ce"], 1e-8)
            if window_norm["conf"] > 0.0 and float(self.args.method_conf_coef) != 0.0:
                eff_conf = float(self.args.method_conf_coef) * float(balance_scales["conf"]) * sample_conf_num / max(window_norm["conf"], 1e-8)

            prompt_token_count = int(batch["prompt_attention_mask"][j].sum().item()) if torch.is_tensor(batch["prompt_attention_mask"]) else int(batch["prompt_used_len"][j])
            gold_token_count = int((batch["gold_labels"][j] != -100).sum().item()) if torch.is_tensor(batch["gold_labels"]) else int(batch["gold_answer_used_len"][j])

            rows.append({
                "trace_version": 1,
                "epoch": int(epoch),
                "train_step": int(next_step),
                "micro_step": int(self.micro_step),
                "window_batch_index": int(batch_index),
                "batch_sample_index": int(j),
                "sample_id": str(batch["id"][j]),
                "data": {
                    "question": self._clip_trace_text(batch["question"][j]),
                    "reference": self._clip_trace_text(batch["reference"][j]),
                    "gold_answer": self._clip_trace_text(batch["answer"][j]),
                },
                "prompt": {
                    "prompt_text": self._clip_trace_text(batch["prompt_text"][j]),
                    "prompt_raw_len": int(batch["prompt_raw_len"][j]),
                    "prompt_used_len": int(batch["prompt_used_len"][j]),
                    "prompt_truncated": bool(batch["prompt_truncated"][j]),
                    "prompt_token_count": int(prompt_token_count),
                    "gold_prompt_raw_len": int(batch["gold_prompt_raw_len"][j]),
                    "gold_prompt_used_len": int(batch["gold_prompt_used_len"][j]),
                    "gold_prompt_truncated": bool(batch["gold_prompt_truncated"][j]),
                    "gold_answer_raw_len": int(batch["gold_answer_raw_len"][j]),
                    "gold_answer_used_len": int(batch["gold_answer_used_len"][j]),
                    "gold_answer_truncated": bool(batch["gold_answer_truncated"][j]),
                    "gold_answer_token_count": int(gold_token_count),
                    "gold_answer_token_ids": [int(x) for x in batch["gold_answer_token_ids"][j]],
                },
                "base_generation": {
                    "base_answer": self._clip_trace_text(base_answers[j]),
                    "base_answer_token_ids": [int(x) for x in base_answer_token_ids[j]],
                    "base_mismatch": bool(metric.get("base_mismatch", False)),
                    "divergence_index": int(metric.get("divergence_index", 0)),
                    "need_bad_branch": bool(metric.get("need_bad_branch", False)),
                },
                "objective": metric,
                "effective_loss": {
                    "sample_ce_term": float(eff_ce),
                    "sample_conf_term": float(eff_conf),
                    "sample_total_term": float(eff_ce + eff_conf),
                    "batch_ce_term": float(ce_term_value),
                    "batch_conf_term": float(conf_term_value),
                    "batch_total_term": float(batch_loss_value),
                },
                "window_context": {
                    "window_norm": {"ce": float(window_norm["ce"]), "conf": float(window_norm["conf"])},
                    "balance_scales": {"ce": float(balance_scales["ce"]), "conf": float(balance_scales["conf"])},
                    "method_sft_coef": float(self.args.method_sft_coef),
                    "method_conf_coef": float(self.args.method_conf_coef),
                },
            })
        return rows

    def _write_sample_trace_step(
        self,
        *,
        step: int,
        epoch: int,
        rows: List[Dict[str, Any]],
        window_stats: List[Dict[str, float]],
        balance_scales: Dict[str, float],
        window_norm: Dict[str, float],
        window_summary: Dict[str, Any],
        grad_before: Dict[str, Any],
        grad_after: Dict[str, Any],
    ):
        if not rows:
            return
        max_per_step = int(getattr(self.args, "sample_trace_max_samples_per_step", 0))
        if max_per_step > 0:
            rows = rows[:max_per_step]
        step_dir = os.path.join(self.args.sample_trace_dir, f"step_{step:06d}")
        ensure_dir(step_dir)
        summary = {
            "epoch": int(epoch),
            "train_step": int(step),
            "num_sample_traces": int(len(rows)),
            "window_stats_prepass": window_stats,
            "window_norm": {"ce": float(window_norm["ce"]), "conf": float(window_norm["conf"])},
            "balance_scales": {"ce": float(balance_scales["ce"]), "conf": float(balance_scales["conf"])},
            "window_summary": self._window_summary_to_python(window_summary),
            "grad_before_clip": grad_before,
            "grad_after_clip": grad_after,
            "formula": self.objective.describe(),
            "sample_files": [],
        }
        for idx, row in enumerate(rows):
            file_token = safe_filename(row.get("sample_id", f"sample_{idx}"), max_len=64)
            filename = f"sample_{idx:04d}__{file_token}.json"
            summary["sample_files"].append(filename)
            row["step_summary"] = {
                "window_summary": summary["window_summary"],
                "grad_before_clip": grad_before,
                "grad_after_clip": grad_after,
            }
            save_json(os.path.join(step_dir, filename), row)
        save_json(os.path.join(step_dir, "summary.json"), summary)

    @torch.inference_mode()

    def evaluate(self):
        from train.eval import evaluate_policy

        self.move_eval_models_to_device()
        self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True
        try:
            step_root = os.path.join(self.args.save_dir, f"eval_step_{self.global_step}")
            ensure_dir(step_root)
            comparisons: List[Dict[str, Any]] = []

            for info in self.eval_job_rows:
                dataset_name = info["dataset_name"]
                split_name = info["split_name"]
                save_dir = os.path.join(step_root, dataset_name, split_name)
                ensure_dir(save_dir)
                comparison = evaluate_policy(
                    eval_data=info["rows"],
                    adapted_model=self.model,
                    adapted_tokenizer=self.tokenizer,
                    base_model=self.base_model,
                    base_tokenizer=self.base_tokenizer,
                    judge_model=self.eval_judge_model,
                    judge_tokenizer=self.eval_judge_tokenizer,
                    max_length=self.args.max_length,
                    max_new_tokens=self.args.max_new_tokens,
                    judge_max_length=self.args.eval_judge_max_length,
                    judge_max_new_tokens=self.args.eval_judge_max_new_tokens,
                    save_dir=save_dir,
                    max_samples=-1,
                    prompt_question_max_chars=self.args.prompt_question_max_chars,
                    prompt_reference_max_chars=self.args.prompt_reference_max_chars,
                    eval_size=self.args.eval_size,
                    debug_max_text_chars=self.args.eval_debug_max_text_chars,
                    eval_step=self.global_step,
                )
                comparison["dataset_name"] = dataset_name
                comparison["split_name"] = split_name
                comparisons.append(comparison)

            aggregate = self._aggregate_eval_comparisons(comparisons)
            save_json(os.path.join(step_root, "comparison.json"), aggregate)

            if self.visual is not None:
                self.visual.log_eval(aggregate)
            self._update_best_checkpoint(aggregate, step_root)
            if getattr(self.args, "save_model_every_eval", False):
                checkpoint_dir = os.path.join(self.args.checkpoint_model_dir, f"eval__step_{int(self.global_step):06d}")
                self.save_model_snapshot(tag="eval", save_dir=checkpoint_dir, extra_info={"eval_step": int(self.global_step)})

            print(
                f"[eval] step={self.global_step} jobs={aggregate['num_jobs']} "
                f"base_h={aggregate['base']['hallucination_rate']:.4f} "
                f"adapted_h={aggregate['adapted']['hallucination_rate']:.4f} "
                f"diff={aggregate['delta']['hallucination_rate']:+.4f} "
                f"adapted_correct={aggregate['adapted'].get('num_correct', 0)}"
            )

            if getattr(self.args, "push_to_hub", False) and getattr(self.args, "hub_repo_id", ""):
                self._push_to_hub(step=self.global_step)

        finally:
            self.offload_eval_models_to_cpu()
            self.model.train()
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

    def _update_usage_counters(self, window_summary: Dict[str, Any]):
        total = int(window_summary.get("num_total_samples", window_summary.get("batch_size", 0)))
        union_used = int(window_summary.get("num_used_union", window_summary.get("num_selected", 0)))
        conf_used = int(window_summary.get("num_used_conf", 0))
        base_mismatch = int(window_summary.get("num_base_mismatch", 0))
        self.running_total_seen += total
        self.running_union_used += union_used
        self.running_conf_used += conf_used
        self.running_base_mismatch += base_mismatch
        self.interval_total_seen += total
        self.interval_union_used += union_used
        self.interval_conf_used += conf_used
        self.interval_base_mismatch += base_mismatch
        self.interval_step_count += 1

    def _build_usage_row(self, *, epoch: int, step: int, interval_steps: int) -> Dict[str, Any]:
        total_interval = max(self.interval_total_seen, 0)
        total_running = max(self.running_total_seen, 0)
        return {
            "epoch": int(epoch),
            "step": int(step),
            "interval_steps": int(interval_steps),
            "interval_union_used": int(self.interval_union_used),
            "interval_conf_used": int(self.interval_conf_used),
            "interval_base_mismatch": int(self.interval_base_mismatch),
            "interval_total_seen": int(total_interval),
            "interval_union_usage_ratio": 0.0 if total_interval == 0 else float(self.interval_union_used) / float(total_interval),
            "interval_conf_usage_ratio": 0.0 if total_interval == 0 else float(self.interval_conf_used) / float(total_interval),
            "interval_base_mismatch_ratio": 0.0 if total_interval == 0 else float(self.interval_base_mismatch) / float(total_interval),
            "running_union_used": int(self.running_union_used),
            "running_conf_used": int(self.running_conf_used),
            "running_base_mismatch": int(self.running_base_mismatch),
            "running_total_seen": int(total_running),
            "running_union_usage_ratio": 0.0 if total_running == 0 else float(self.running_union_used) / float(total_running),
            "running_conf_usage_ratio": 0.0 if total_running == 0 else float(self.running_conf_used) / float(total_running),
            "running_base_mismatch_ratio": 0.0 if total_running == 0 else float(self.running_base_mismatch) / float(total_running),
        }

    def _flush_usage_interval(self, *, epoch: int, force: bool = False):
        if getattr(self.args, "train_pipeline", "new_method") != "new_method":
            return
        every = int(getattr(self.args, "usage_log_every", 50))
        if every <= 0:
            return
        if self.interval_step_count <= 0:
            return
        if (not force) and (self.global_step % every != 0):
            return
        row = self._build_usage_row(epoch=epoch, step=self.global_step, interval_steps=self.interval_step_count)
        append_jsonl(self.usage_summary_path, row)
        if self.visual is not None:
            self.visual.log_usage(row)
        print(
            f"[usage] step={self.global_step} last_{self.interval_step_count}_steps "
            f"union={row['interval_union_used']}/{row['interval_total_seen']} ({row['interval_union_usage_ratio']:.4f}) "
            f"conf={row['interval_conf_used']}/{row['interval_total_seen']} ({row['interval_conf_usage_ratio']:.4f}) "
            f"mismatch={row['interval_base_mismatch']}/{row['interval_total_seen']} ({row['interval_base_mismatch_ratio']:.4f}) "
            f"running_union={row['running_union_used']}/{row['running_total_seen']} ({row['running_union_usage_ratio']:.4f})"
        )
        self.interval_total_seen = 0
        self.interval_union_used = 0
        self.interval_conf_used = 0
        self.interval_base_mismatch = 0
        self.interval_step_count = 0

    def _finalize_usage_tracking(self, *, epoch: int):
        if getattr(self.args, "train_pipeline", "new_method") != "new_method":
            return
        self._flush_usage_interval(epoch=epoch, force=True)
        save_json(
            self.usage_running_path,
            {
                "step": int(self.global_step),
                "running_union_used": int(self.running_union_used),
                "running_conf_used": int(self.running_conf_used),
                "running_base_mismatch": int(self.running_base_mismatch),
                "running_total_seen": int(self.running_total_seen),
                "running_union_usage_ratio": 0.0 if self.running_total_seen == 0 else float(self.running_union_used) / float(self.running_total_seen),
                "running_conf_usage_ratio": 0.0 if self.running_total_seen == 0 else float(self.running_conf_used) / float(self.running_total_seen),
                "running_base_mismatch_ratio": 0.0 if self.running_total_seen == 0 else float(self.running_base_mismatch) / float(self.running_total_seen),
            },
        )

    def train_new_method(self):
        self.model.train()
        stop = False
        last_epoch = 0
        for epoch in range(1, self.args.num_epochs + 1):
            last_epoch = epoch
            train_iter = iter(self.train_loader)
            while True:
                window_batches = []
                for _ in range(self.args.gradient_accumulation_steps):
                    try:
                        window_batches.append(next(train_iter))
                    except StopIteration:
                        break
                if not window_batches:
                    break

                next_step = self.global_step + 1
                write_trace = self._trace_enabled_for_step(next_step)
                base_cache = self.precompute_base_answers(window_batches)
                window_stats = []

                self.model.eval()
                with torch.inference_mode():
                    for batch, base_pack in zip(window_batches, base_cache):
                        stat_dict = self.objective(
                            batch,
                            base_answers=base_pack[0],
                            base_answer_token_ids=base_pack[1],
                            balance_scales=None,
                            update_mag_stats=False,
                        )
                        window_stats.append({
                            "ce_num": float(stat_dict["ce_num"]),
                            "ce_denom": float(stat_dict["ce_denom"]),
                            "conf_num": float(stat_dict["conf_num"]),
                            "conf_denom": float(stat_dict["conf_denom"]),
                            "raw_mag_mean": float(stat_dict["raw_mag_mean"].detach().float().item()),
                            "mag_mean": float(stat_dict["mag_mean"].detach().float().item()),
                            "mag_pos_mean": float(stat_dict["mag_pos_mean"].detach().float().item()),
                            "mag_neg_mean": float(stat_dict["mag_neg_mean"].detach().float().item()),
                            "rank_loss_mean": float(stat_dict["rank_loss_mean"].detach().float().item()),
                            "num_halluci": int(stat_dict["num_halluci"]),
                            "num_clean": int(stat_dict["num_clean"]),
                            "num_selected": int(stat_dict["num_selected"]),
                            "num_used_ce": int(stat_dict.get("num_used_ce", stat_dict["num_selected"])),
                            "num_used_conf": int(stat_dict.get("num_used_conf", 0)),
                            "num_used_union": int(stat_dict.get("num_used_union", stat_dict["num_selected"])),
                            "num_conf_eligible": int(stat_dict.get("num_conf_eligible", 0)),
                            "num_base_mismatch": int(stat_dict.get("num_base_mismatch", 0)),
                            "batch_size": int(stat_dict["batch_size"]),
                        })
                self.model.train()

                balance_scales = self.compute_window_balance(window_stats)
                window_norm = self.build_window_normalizers(window_stats)
                window_summary = self.summarize_window(window_stats, balance_scales)

                self.optimizer.zero_grad(set_to_none=True)
                has_any_grad = False
                step_sample_traces: List[Dict[str, Any]] = []

                for batch_index, (batch, base_pack) in enumerate(zip(window_batches, base_cache)):
                    self.micro_step += 1
                    loss_dict = self.objective(
                        batch,
                        base_answers=base_pack[0],
                        base_answer_token_ids=base_pack[1],
                        balance_scales=None,
                        update_mag_stats=False,
                    )

                    batch_loss = None
                    ce_term_value = 0.0
                    conf_term_value = 0.0
                    if window_norm["ce"] > 0.0 and float(self.args.method_sft_coef) != 0.0:
                        term = (
                            float(self.args.method_sft_coef)
                            * float(balance_scales["ce"])
                            * loss_dict["ce_num"]
                            / max(window_norm["ce"], 1e-8)
                        )
                        ce_term_value = float(term.detach().float().item())
                        batch_loss = term if batch_loss is None else batch_loss + term

                    if window_norm["conf"] > 0.0 and float(self.args.method_conf_coef) != 0.0:
                        term = (
                            float(self.args.method_conf_coef)
                            * float(balance_scales["conf"])
                            * loss_dict["conf_num"]
                            / max(window_norm["conf"], 1e-8)
                        )
                        conf_term_value = float(term.detach().float().item())
                        batch_loss = term if batch_loss is None else batch_loss + term

                    batch_loss_value = ce_term_value + conf_term_value
                    if write_trace:
                        step_sample_traces.extend(
                            self._build_sample_trace_rows(
                                epoch=epoch,
                                next_step=next_step,
                                batch_index=batch_index,
                                batch=batch,
                                base_answers=base_pack[0],
                                base_answer_token_ids=base_pack[1],
                                loss_dict=loss_dict,
                                ce_term_value=ce_term_value,
                                conf_term_value=conf_term_value,
                                batch_loss_value=batch_loss_value,
                                balance_scales=balance_scales,
                                window_norm=window_norm,
                            )
                        )

                    if batch_loss is None:
                        continue
                    if torch.is_tensor(batch_loss) and not batch_loss.requires_grad:
                        continue

                    batch_loss.backward()
                    has_any_grad = True

                if not has_any_grad:
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                grad_before = compute_grad_stats(self.trainable_params)
                if self.args.max_grad_norm and self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, self.args.max_grad_norm)
                grad_after = compute_grad_stats(self.trainable_params)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                self._update_usage_counters(window_summary)
                window_summary["running_union_used"] = int(self.running_union_used)
                window_summary["running_conf_used"] = int(self.running_conf_used)
                window_summary["running_total_seen"] = int(self.running_total_seen)

                if write_trace:
                    self._write_sample_trace_step(
                        step=self.global_step,
                        epoch=epoch,
                        rows=step_sample_traces,
                        window_stats=window_stats,
                        balance_scales=balance_scales,
                        window_norm=window_norm,
                        window_summary=window_summary,
                        grad_before=grad_before,
                        grad_after=grad_after,
                    )

                if self.global_step % self.args.log_every == 0:
                    print(format_train_step_debug(epoch, self.global_step, window_summary))

                if self.visual is not None and self.global_step % self.args.visual_every == 0:
                    row = {
                        "epoch": epoch,
                        "step": self.global_step,
                        "loss": float(window_summary["loss"].detach().float().item()),
                        "ce_base_loss": float(window_summary["ce_base_loss"].detach().float().item()),
                        "conf_loss": float(window_summary["conf_loss"].detach().float().item()),
                        "ce_contrib": float(window_summary["ce_contrib"].detach().float().item()),
                        "conf_contrib": float(window_summary["conf_contrib"].detach().float().item()),
                        "raw_mag_mean": float(window_summary["raw_mag_mean"].detach().float().item()),
                        "mag_mean": float(window_summary["mag_mean"].detach().float().item()),
                        "mag_pos_mean": float(window_summary["mag_pos_mean"].detach().float().item()),
                        "mag_neg_mean": float(window_summary["mag_neg_mean"].detach().float().item()),
                        "rank_loss_mean": float(window_summary["rank_loss_mean"].detach().float().item()),
                        "scale_ce": float(window_summary["scale_ce"]),
                        "scale_conf": float(window_summary["scale_conf"]),
                        "num_halluci": int(window_summary["num_halluci"]),
                        "num_selected": int(window_summary["num_selected"]),
                        "num_used_ce": int(window_summary["num_used_ce"]),
                        "num_used_conf": int(window_summary["num_used_conf"]),
                        "num_used_union": int(window_summary["num_used_union"]),
                        "num_conf_eligible": int(window_summary["num_conf_eligible"]),
                        "num_base_mismatch": int(window_summary["num_base_mismatch"]),
                        "num_total_samples": int(window_summary["num_total_samples"]),
                        "running_union_used": int(self.running_union_used),
                        "running_conf_used": int(self.running_conf_used),
                        "running_total_seen": int(self.running_total_seen),
                        "grad_before_clip": grad_before,
                        "grad_after_clip": grad_after,
                    }
                    self.visual.log_train(row)

                self._flush_usage_interval(epoch=epoch)

                if self.args.eval_every > 0 and self.global_step % self.args.eval_every == 0:
                    self.evaluate()

                if self.args.early_stop_step > 0 and self.global_step >= self.args.early_stop_step:
                    if not (self.args.eval_every > 0 and self.global_step % self.args.eval_every == 0):
                        self.evaluate()
                    stop = True
                    break

                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    stop = True
                    break

            if stop:
                break

        self._finalize_usage_tracking(epoch=last_epoch)
        if self.args.eval_every <= 0 or self.global_step % self.args.eval_every != 0:
            self.evaluate()

    def train_sft_only(self):
        self.model.train()
        stop = False
        last_epoch = 0
        for epoch in range(1, self.args.num_epochs + 1):
            last_epoch = epoch
            train_iter = iter(self.train_loader)
            while True:
                window_batches = []
                for _ in range(self.args.gradient_accumulation_steps):
                    try:
                        window_batches.append(next(train_iter))
                    except StopIteration:
                        break
                if not window_batches:
                    break

                self.model.eval()
                window_stats = []
                with torch.inference_mode():
                    for batch in window_batches:
                        stat_dict = self.objective.forward_sft_only(batch)
                        window_stats.append({
                            "ce_num": float(stat_dict["ce_num"]),
                            "ce_denom": float(stat_dict["ce_denom"]),
                            "conf_num": 0.0,
                            "conf_denom": 0.0,
                            "raw_mag_mean": 0.0,
                            "mag_mean": 0.0,
                            "mag_pos_mean": 0.0,
                            "mag_neg_mean": 0.0,
                            "rank_loss_mean": 0.0,
                            "num_halluci": 0,
                            "num_clean": int(stat_dict["batch_size"]),
                            "num_selected": int(stat_dict["batch_size"]),
                            "num_used_ce": int(stat_dict["batch_size"]),
                            "num_used_conf": 0,
                            "num_used_union": int(stat_dict["batch_size"]),
                            "num_conf_eligible": 0,
                            "num_base_mismatch": 0,
                            "batch_size": int(stat_dict["batch_size"]),
                        })
                self.model.train()

                balance_scales = {"ce": 1.0, "conf": 0.0}
                window_norm = {"ce": max(sum(float(s.get("ce_denom", 0.0)) for s in window_stats), 0.0), "conf": 0.0}
                window_summary = self.summarize_window(window_stats, balance_scales)

                self.optimizer.zero_grad(set_to_none=True)
                has_any_grad = False
                for batch in window_batches:
                    self.micro_step += 1
                    loss_dict = self.objective.forward_sft_only(batch)
                    if window_norm["ce"] <= 0.0:
                        continue
                    batch_loss = loss_dict["ce_num"] / max(window_norm["ce"], 1e-8)
                    if torch.is_tensor(batch_loss) and batch_loss.requires_grad:
                        batch_loss.backward()
                        has_any_grad = True

                if not has_any_grad:
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                grad_before = compute_grad_stats(self.trainable_params)
                if self.args.max_grad_norm and self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, self.args.max_grad_norm)
                grad_after = compute_grad_stats(self.trainable_params)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                if self.global_step % self.args.log_every == 0:
                    print(format_train_step_debug(epoch, self.global_step, window_summary))

                if self.visual is not None and self.global_step % self.args.visual_every == 0:
                    row = {
                        "epoch": epoch,
                        "step": self.global_step,
                        "loss": float(window_summary["loss"].detach().float().item()),
                        "ce_base_loss": float(window_summary["ce_base_loss"].detach().float().item()),
                        "conf_loss": 0.0,
                        "ce_contrib": float(window_summary["ce_contrib"].detach().float().item()),
                        "conf_contrib": 0.0,
                        "raw_mag_mean": 0.0,
                        "mag_mean": 0.0,
                        "mag_pos_mean": 0.0,
                        "mag_neg_mean": 0.0,
                        "rank_loss_mean": 0.0,
                        "scale_ce": 1.0,
                        "scale_conf": 0.0,
                        "num_halluci": 0,
                        "num_selected": int(window_summary["num_selected"]),
                        "num_used_ce": int(window_summary["num_used_ce"]),
                        "num_used_conf": 0,
                        "num_used_union": int(window_summary["num_used_union"]),
                        "num_conf_eligible": 0,
                        "num_base_mismatch": 0,
                        "num_total_samples": int(window_summary["num_total_samples"]),
                        "grad_before_clip": grad_before,
                        "grad_after_clip": grad_after,
                    }
                    self.visual.log_train(row)

                if self.args.eval_every > 0 and self.global_step % self.args.eval_every == 0:
                    self.evaluate()

                if self.args.early_stop_step > 0 and self.global_step >= self.args.early_stop_step:
                    if not (self.args.eval_every > 0 and self.global_step % self.args.eval_every == 0):
                        self.evaluate()
                    stop = True
                    break

                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    stop = True
                    break

            if stop:
                break

        if self.args.eval_every <= 0 or self.global_step % self.args.eval_every != 0:
            self.evaluate()

    def train(self):
        if getattr(self.args, "train_pipeline", "new_method") == "sft_only":
            return self.train_sft_only()
        return self.train_new_method()
