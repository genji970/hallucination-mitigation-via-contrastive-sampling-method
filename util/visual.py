import json
import os
from typing import Any, Dict, List

import torch


def to_python(value: Any):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(v) for v in value]
    return value


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_python(obj), f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, row: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_python(row), ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return default if x is None else float(x)
    except Exception:
        return default


def build_param_report(model, objective_module=None, include_details: bool = False) -> Dict[str, Any]:
    total = 0
    trainable = 0
    details = []
    for name, p in model.named_parameters():
        numel = int(p.numel())
        total += numel
        if p.requires_grad:
            trainable += numel
        if include_details:
            details.append({"name": name, "numel": numel, "shape": list(p.shape), "trainable": bool(p.requires_grad)})
    report = {
        "model": {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_ratio": float(trainable / max(total, 1)),
        },
        "objective": {
            "layer_idx": int(getattr(objective_module, "layer_idx", -1)) if objective_module is not None else -1,
            "hidden_size": int(getattr(objective_module, "hidden_size", 0)) if objective_module is not None else 0,
            "row_mask_active": int(getattr(objective_module, "row_mask", torch.zeros(1)).sum().item()) if objective_module is not None and hasattr(objective_module, "row_mask") else 0,
            "row_mask_total": int(getattr(objective_module, "row_mask", torch.zeros(1)).numel()) if objective_module is not None and hasattr(objective_module, "row_mask") else 0,
            "row_mask_summary": getattr(objective_module, "row_mask_summary", []) if objective_module is not None else [],
        },
    }
    if include_details:
        report["model"]["details"] = details
    return report


def format_param_report(report: Dict[str, Any], show_details: bool = False) -> str:
    model = report["model"]
    obj = report.get("objective", {})
    lines = [
        "[param] "
        f"model_trainable={model['trainable_params']}/{model['total_params']} "
        f"({model['trainable_ratio']:.4%})"
        + (f" | layer={obj.get('layer_idx', -1)} trainable_rows={obj.get('row_mask_active', 0)}/{obj.get('row_mask_total', 0)}" if obj else "")
    ]
    if show_details and report["model"].get("details"):
        for item in report["model"]["details"]:
            if item["trainable"]:
                lines.append(f"  [trainable] {item['name']} shape={item['shape']} numel={item['numel']}")
    return "\n".join(lines)


class VisualLogger:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        ensure_dir(self.root_dir)
        self.train_metrics_path = os.path.join(self.root_dir, "train_metrics.jsonl")
        self.eval_metrics_path = os.path.join(self.root_dir, "eval_metrics.jsonl")
        self.param_report_path = os.path.join(self.root_dir, "param_report.json")
        self.run_args_path = os.path.join(self.root_dir, "run_args.json")
        self.usage_metrics_path = os.path.join(self.root_dir, "usage_metrics.jsonl")
        self.train_curve_path = os.path.join(self.root_dir, "train_curve.png")
        self.eval_curve_path = os.path.join(self.root_dir, "eval_curve.png")
        self.usage_curve_path = os.path.join(self.root_dir, "usage_curve.png")
        self.eval_counts_curve_path = os.path.join(self.root_dir, "eval_counts_curve.png")

    def log_args(self, args):
        save_json(self.run_args_path, vars(args))

    def log_param_report(self, report: Dict[str, Any]):
        save_json(self.param_report_path, report)

    def log_train(self, row: Dict[str, Any]):
        append_jsonl(self.train_metrics_path, row)
        self.render_train_plot()

    def log_eval(self, row: Dict[str, Any]):
        append_jsonl(self.eval_metrics_path, row)
        self.render_eval_plot()

    def log_usage(self, row: Dict[str, Any]):
        append_jsonl(self.usage_metrics_path, row)
        self.render_usage_plot()

    def render_train_plot(self):
        rows = read_jsonl(self.train_metrics_path)
        if not rows:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [safe_float(r.get("step", i + 1), i + 1) for i, r in enumerate(rows)]
        total = [safe_float(r.get("loss", 0.0)) for r in rows]
        ce = [safe_float(r.get("ce_base_loss", 0.0)) for r in rows]
        conf = [safe_float(r.get("conf_loss", 0.0)) for r in rows]
        fig = plt.figure(figsize=(10, 6))
        plt.plot(steps, total, label="total")
        plt.plot(steps, ce, label="ce_base")
        plt.plot(steps, conf, label="conf")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training losses")
        plt.legend()
        plt.tight_layout()
        fig.savefig(self.train_curve_path)
        plt.close(fig)

    def render_eval_plot(self):
        rows = read_jsonl(self.eval_metrics_path)
        if not rows:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [safe_float(r.get("eval_step", r.get("step", i + 1)), i + 1) for i, r in enumerate(rows)]
        base_h = [safe_float(r.get("base", {}).get("hallucination_rate", 0.0)) for r in rows]
        adapted_h = [safe_float(r.get("adapted", {}).get("hallucination_rate", 0.0)) for r in rows]
        fig = plt.figure(figsize=(10, 6))
        plt.plot(steps, base_h, marker="o", label="base_h")
        plt.plot(steps, adapted_h, marker="o", label="adapted_h")
        plt.xlabel("eval step")
        plt.ylabel("hallucination rate")
        plt.title("Eval hallucination rate")
        plt.legend()
        plt.tight_layout()
        fig.savefig(self.eval_curve_path)
        plt.close(fig)

        base_correct = [
            safe_float(
                r.get("base", {}).get(
                    "num_correct",
                    safe_float(r.get("base", {}).get("num_samples", 0.0)) - safe_float(r.get("base", {}).get("num_hallucinated", 0.0)),
                )
            )
            for r in rows
        ]
        adapted_correct = [
            safe_float(
                r.get("adapted", {}).get(
                    "num_correct",
                    safe_float(r.get("adapted", {}).get("num_samples", 0.0)) - safe_float(r.get("adapted", {}).get("num_hallucinated", 0.0)),
                )
            )
            for r in rows
        ]
        improved = [safe_float(r.get("label_diff", {}).get("improved", 0.0)) for r in rows]
        worsened = [safe_float(r.get("label_diff", {}).get("worsened", 0.0)) for r in rows]
        fig = plt.figure(figsize=(10, 6))
        plt.plot(steps, base_correct, marker="o", label="base_correct")
        plt.plot(steps, adapted_correct, marker="o", label="adapted_correct")
        plt.plot(steps, improved, marker="o", label="improved")
        plt.plot(steps, worsened, marker="o", label="worsened")
        plt.xlabel("eval step")
        plt.ylabel("count")
        plt.title("Eval comparison counts")
        plt.legend()
        plt.tight_layout()
        fig.savefig(self.eval_counts_curve_path)
        plt.close(fig)

    def render_usage_plot(self):
        rows = read_jsonl(self.usage_metrics_path)
        if not rows:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [safe_float(r.get("step", i + 1), i + 1) for i, r in enumerate(rows)]
        interval_ratio = [safe_float(r.get("interval_union_usage_ratio", 0.0)) for r in rows]
        running_ratio = [safe_float(r.get("running_union_usage_ratio", 0.0)) for r in rows]
        conf_ratio = [safe_float(r.get("interval_conf_usage_ratio", 0.0)) for r in rows]
        fig = plt.figure(figsize=(10, 6))
        plt.plot(steps, interval_ratio, label="interval_union")
        plt.plot(steps, running_ratio, label="running_union")
        plt.plot(steps, conf_ratio, label="interval_conf")
        plt.xlabel("step")
        plt.ylabel("usage ratio")
        plt.title("New-method sample usage")
        plt.legend()
        plt.tight_layout()
        fig.savefig(self.usage_curve_path)
        plt.close(fig)
