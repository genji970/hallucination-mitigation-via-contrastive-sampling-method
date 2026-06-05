import math
import re
from typing import Any, Dict, Iterable, List

import torch


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    return " ".join(text.strip().lower().split())


def try_parse_number(text: Any):
    text = normalize_text(text)
    if text == "":
        return None
    text = text.replace(",", "")
    m = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(text)
    except Exception:
        return None


def answers_match(gold: Any, pred: Any, mode: str = "numeric_or_exact") -> bool:
    g = normalize_text(gold)
    p = normalize_text(pred)
    if mode == "exact":
        return g == p
    gn = try_parse_number(g)
    pn = try_parse_number(p)
    if gn is not None and pn is not None:
        return abs(gn - pn) < 1e-9
    return g == p


def get_eval_uid(ex: Dict[str, Any], i: int) -> str:
    if ex.get("eval_uid") is not None:
        return str(ex.get("eval_uid"))
    return f"{i:08d}::{str(ex.get('id', i))}"


def to_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return 0.0
        return float(x.detach().float().mean().item()) if x.numel() > 1 else float(x.detach().float().item())
    if x is None:
        return 0.0
    return float(x)


def ratio(numer: float, denom: float) -> float:
    return 0.0 if denom == 0 else float(numer) / float(denom)


def compute_grad_stats(parameters: Iterable[torch.nn.Parameter]) -> Dict[str, float]:
    total_sq = 0.0
    max_abs = 0.0
    params_with_grad = 0
    elems_with_grad = 0
    nonzero_grad_elems = 0
    for p in parameters:
        grad = getattr(p, "grad", None)
        if grad is None:
            continue
        g = grad.detach().float()
        params_with_grad += 1
        elems_with_grad += g.numel()
        total_sq += float(torch.sum(g * g).item())
        if g.numel() > 0:
            max_abs = max(max_abs, float(g.abs().max().item()))
            nonzero_grad_elems += int((g != 0).sum().item())
    return {
        "grad_l2": math.sqrt(total_sq),
        "grad_max_abs": max_abs,
        "params_with_grad": params_with_grad,
        "elems_with_grad": elems_with_grad,
        "nonzero_grad_elems": nonzero_grad_elems,
        "nonzero_grad_frac": ratio(nonzero_grad_elems, elems_with_grad),
    }


def format_train_step_debug(epoch: int, step: int, loss_dict: Dict[str, Any]) -> str:
    total = int(loss_dict.get("num_total_samples", loss_dict.get("batch_size", 0)))
    used_ce = int(loss_dict.get("num_used_ce", loss_dict.get("num_selected", 0)))
    used_conf = int(loss_dict.get("num_used_conf", loss_dict.get("num_conf_eligible", 0)))
    used_union = int(loss_dict.get("num_used_union", used_ce))
    base_mismatch = int(loss_dict.get("num_base_mismatch", 0))
    pieces = [
        f"[train] epoch={epoch} step={step}",
        f"loss={to_float(loss_dict.get('loss')):.6f}",
        f"ce_base={to_float(loss_dict.get('ce_base_loss')):.6f}",
        f"conf={to_float(loss_dict.get('conf_loss')):.6f}",
        "|",
        f"ce_eff={to_float(loss_dict.get('ce_contrib')):.6f}",
        f"conf_eff={to_float(loss_dict.get('conf_contrib')):.6f}",
        "|",
        f"ce_used={used_ce}/{total}",
        f"conf_used={used_conf}/{total}",
        f"union_used={used_union}/{total}",
    ]
    if base_mismatch > 0 or "num_base_mismatch" in loss_dict:
        pieces.append(f"base_mismatch={base_mismatch}/{total}")
    pieces.extend([
        f"raw_mag_mean={to_float(loss_dict.get('raw_mag_mean')):.6f}",
        f"mag_mean={to_float(loss_dict.get('mag_mean')):.6f}",
        f"mag_pos_mean={to_float(loss_dict.get('mag_pos_mean')):.6f}",
        f"mag_neg_mean={to_float(loss_dict.get('mag_neg_mean')):.6f}",
        f"rank_mean={to_float(loss_dict.get('rank_loss_mean')):.6f}",
        "|",
        f"scale_ce={to_float(loss_dict.get('scale_ce')):.3f}",
        f"scale_conf={to_float(loss_dict.get('scale_conf')):.3f}",
    ])
    if "running_union_used" in loss_dict and "running_total_seen" in loss_dict:
        pieces.extend([
            "|",
            f"running_union={int(loss_dict.get('running_union_used', 0))}/{int(loss_dict.get('running_total_seen', 0))}",
            f"running_conf={int(loss_dict.get('running_conf_used', 0))}/{int(loss_dict.get('running_total_seen', 0))}",
        ])
    return " ".join(pieces)
def limit_text(text: Any, max_chars: int = 0) -> str:
    text = "" if text is None else str(text)
    if max_chars is None or int(max_chars) <= 0:
        return text
    return text[: int(max_chars)]


def safe_filename(text: Any, max_len: int = 80) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = text.strip("._-")
    if not text:
        text = "sample"
    return text[: max(8, int(max_len))]



def summarize_label_status(label: Any) -> str:
    if label is None:
        return "unknown"
    return str(label)
