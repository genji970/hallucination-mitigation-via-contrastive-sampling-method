import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import torch

from util.util import get_eval_uid

LABEL_RE = re.compile(r"\b(?:NOT_HALLUCINATED|HALLUCINATED)\b", re.IGNORECASE)
BASE_PRED_CACHE: Dict[str, Any] = {}
JUDGE_RESULT_CACHE: Dict[str, Dict[str, Any]] = {}


def write_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def limit_text(text: Any, max_chars: int) -> str:
    text = "" if text is None else str(text)
    return text[:max_chars] if max_chars and max_chars > 0 else text


def normalize_example(ex: Dict[str, Any], i: int) -> Dict[str, Any]:
    return {
        "id": str(ex.get("id", i)),
        "eval_uid": get_eval_uid(ex, i),
        "raw_index": int(ex.get("raw_index", i)),
        "question": "" if ex.get("question") is None else str(ex.get("question")),
        "reference": "" if ex.get("reference") is None else str(ex.get("reference")),
        "answer": "" if ex.get("answer") is None else str(ex.get("answer")),
    }


def data_hash(data: List[Dict[str, Any]]) -> str:
    parts = []
    for i, ex in enumerate(data):
        parts.append(f"{get_eval_uid(ex, i)}|{ex.get('id', i)}|{ex.get('question', '')}|{ex.get('reference', '')}|{ex.get('answer', '')}")
    return hashlib.md5("\n".join(parts).encode("utf-8")).hexdigest()


def prediction_cache_key(data: List[Dict[str, Any]], *, model_name_or_path: str, max_length: int, max_new_tokens: int, prompt_question_max_chars: int, prompt_reference_max_chars: int) -> str:
    return "|".join([
        data_hash(data),
        str(model_name_or_path),
        str(max_length),
        str(max_new_tokens),
        str(prompt_question_max_chars),
        str(prompt_reference_max_chars),
    ])


def judge_cache_key(ex: Dict[str, Any], *, judge_model_name_or_path: str, judge_max_length: int, judge_max_new_tokens: int) -> str:
    return "|".join([
        str(judge_model_name_or_path),
        str(judge_max_length),
        str(judge_max_new_tokens),
        str(ex.get("question", "")),
        str(ex.get("reference", "")),
        str(ex.get("answer", "")),
        str(ex.get("prediction", "")),
    ])


def summarize_metrics(successes: List[Dict[str, Any]]) -> Dict[str, float]:
    n_total = len(successes)
    n_h = sum(1 for row in successes if row.get("label") == "HALLUCINATED")
    return {
        "num_samples": n_total,
        "num_hallucinated": n_h,
        "hallucination_rate": float(n_h / max(n_total, 1)),
    }


def check_id_alignment(expected_rows: List[Dict[str, Any]], predicted_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    expected = [str(row.get("eval_uid", "")) for row in expected_rows]
    predicted = [str(row.get("eval_uid", "")) for row in predicted_rows]
    aligned = expected == predicted and len(expected) == len(predicted)
    out = {
        "aligned": aligned,
        "expected_len": len(expected),
        "predicted_len": len(predicted),
    }
    if not aligned:
        out["first_expected"] = expected[:5]
        out["first_predicted"] = predicted[:5]
    return out


def render_prompt(ex: Dict[str, Any], *, tokenizer=None, question_max_chars: int = 0, reference_max_chars: int = 0) -> str:
    question = "" if ex.get("question") is None else str(ex.get("question"))
    reference = "" if ex.get("reference") is None else str(ex.get("reference"))
    if question_max_chars and question_max_chars > 0:
        question = question[:question_max_chars]
    if reference_max_chars and reference_max_chars > 0:
        reference = reference[:reference_max_chars]
    system_text = (
        "You are a precise question-answering assistant. "
        "Answer the user's question using only the provided reference. "
        "Do not explain your reasoning. Do not restate the question. "
        "Return only the final answer text. If the answer is a number, return only the number."
    )
    user_text = f"REFERENCE:\n{reference}\n\nQUESTION:\n{question}" if reference.strip() else f"QUESTION:\n{question}"
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def render_judge_prompt(ex: Dict[str, Any]) -> str:
    question = str(ex.get("question", "")).strip()
    reference = str(ex.get("reference", "")).strip()
    gold_answer = str(ex.get("answer", "")).strip()
    prediction = str(ex.get("prediction", "")).strip()
    parts = [
        "You are a strict hallucination judge.",
        "Given QUESTION, GOLD_ANSWER, optional REFERENCE, and MODEL_PREDICTION, decide whether MODEL_PREDICTION is hallucinated.",
        "Return NOT_HALLUCINATED if the prediction gives the same final answer as GOLD_ANSWER, allowing harmless paraphrases.",
        "Return HALLUCINATED if the prediction contradicts GOLD_ANSWER, fails to answer the question, or adds incorrect factual claims.",
        "Output only one label: NOT_HALLUCINATED or HALLUCINATED.",
        "",
        f"QUESTION:\n{question}",
        "",
        f"GOLD_ANSWER:\n{gold_answer}",
    ]
    if reference:
        parts.extend(["", f"REFERENCE:\n{reference}"])
    parts.extend(["", f"MODEL_PREDICTION:\n{prediction}", "", "LABEL:"])
    return "\n".join(parts)


def parse_judge_label(text: Any) -> Optional[str]:
    text = "" if text is None else str(text)
    matches = [m.upper() for m in LABEL_RE.findall(text)]
    if not matches:
        return None
    uniq = []
    for m in matches:
        if m not in uniq:
            uniq.append(m)
    return uniq[0] if len(uniq) == 1 else None


@torch.inference_mode()
def judge_examples_model(examples: List[Dict[str, Any]], *, judge_model, judge_tokenizer, judge_max_length: int, judge_max_new_tokens: int, size: int = 8, debug_max_text_chars: int = 0):
    successes, failures = [], []
    if not examples:
        return successes, failures

    judge_model_name = getattr(judge_model, "name_or_path", getattr(judge_model.config, "_name_or_path", "judge"))
    pending_rows: List[Dict[str, Any]] = []
    unique_prompts: List[str] = []
    unique_prompt_token_lengths_raw: List[int] = []
    unique_examples: List[Dict[str, Any]] = []
    unique_keys: List[str] = []
    pending_key_to_slots: Dict[str, List[Dict[str, Any]]] = {}

    def append_cached_result(ex: Dict[str, Any], idx: int, cached: Dict[str, Any]):
        row = dict(cached)
        row["id"] = str(ex.get("id", idx))
        row["eval_uid"] = get_eval_uid(ex, idx)
        if row.get("status") is None:
            successes.append(row)
        else:
            failures.append(row)

    for idx, ex in enumerate(examples):
        pred = str(ex.get("prediction", "") or "").strip()
        gold = str(ex.get("answer", "") or "").strip()
        if not pred:
            failures.append({"id": str(ex.get("id", idx)), "eval_uid": get_eval_uid(ex, idx), "status": "no_prediction"})
            continue
        if not gold:
            failures.append({"id": str(ex.get("id", idx)), "eval_uid": get_eval_uid(ex, idx), "status": "no_gold_answer"})
            continue
        key = judge_cache_key(
            ex,
            judge_model_name_or_path=judge_model_name,
            judge_max_length=judge_max_length,
            judge_max_new_tokens=judge_max_new_tokens,
        )
        cached = JUDGE_RESULT_CACHE.get(key)
        if cached is not None:
            append_cached_result(ex, idx, cached)
            continue
        slot = {"example": ex, "idx": idx}
        if key in pending_key_to_slots:
            pending_key_to_slots[key].append(slot)
            continue
        prompt = render_judge_prompt(ex)
        pending_key_to_slots[key] = [slot]
        pending_rows.append(slot)
        unique_keys.append(key)
        unique_prompts.append(prompt)
        unique_examples.append(ex)
        unique_prompt_token_lengths_raw.append(len(judge_tokenizer(prompt, add_special_tokens=False)["input_ids"]))

    if not unique_prompts:
        return successes, failures

    orig_side = judge_tokenizer.padding_side
    orig_trunc_side = getattr(judge_tokenizer, "truncation_side", "right")
    judge_tokenizer.padding_side = "left"
    judge_tokenizer.truncation_side = "left"
    if judge_tokenizer.pad_token_id is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token
    was_training = judge_model.training
    judge_model.eval()

    try:
        for start in range(0, len(unique_prompts), size):
            prompts_part = unique_prompts[start:start + size]
            examples_part = unique_examples[start:start + size]
            keys_part = unique_keys[start:start + size]
            tok_kw = {"return_tensors": "pt", "padding": True, "add_special_tokens": False}
            if judge_max_length and judge_max_length > 0:
                tok_kw["truncation"] = True
                tok_kw["max_length"] = judge_max_length
            else:
                tok_kw["truncation"] = False
            enc = judge_tokenizer(prompts_part, **tok_kw).to(judge_model.device)
            out = judge_model.generate(
                **enc,
                max_new_tokens=judge_max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=judge_tokenizer.pad_token_id,
                eos_token_id=judge_tokenizer.eos_token_id,
            )
            input_len = enc["input_ids"].shape[1]
            for i, ex in enumerate(examples_part):
                raw_output = judge_tokenizer.decode(out[i][input_len:], skip_special_tokens=True).strip()
                base_row: Dict[str, Any]
                label = parse_judge_label(raw_output)
                if label is None:
                    base_row = {
                        "status": "parse_failed",
                        "raw_output": limit_text(raw_output, debug_max_text_chars),
                    }
                else:
                    base_row = {
                        "label": label,
                        "raw_output": limit_text(raw_output, debug_max_text_chars),
                        "judge_prompt_text": limit_text(prompts_part[i], debug_max_text_chars),
                        "debug_judge_prompt_tokens_raw": unique_prompt_token_lengths_raw[start + i],
                        "debug_judge_prompt_tokens_used": int(enc["attention_mask"][i].sum().item()),
                    }
                key = keys_part[i]
                JUDGE_RESULT_CACHE[key] = dict(base_row)
                for slot in pending_key_to_slots.get(key, []):
                    append_cached_result(slot["example"], slot["idx"], base_row)
    finally:
        judge_tokenizer.padding_side = orig_side
        judge_tokenizer.truncation_side = orig_trunc_side
        if was_training:
            judge_model.train()
    return successes, failures


@torch.inference_mode()
def generate_predictions(examples: List[Dict[str, Any]], *, answer_model, answer_tokenizer, max_length: int, max_new_tokens: int, prompt_question_max_chars: int = 0, prompt_reference_max_chars: int = 0, size: int = 8):
    normalized = [normalize_example(ex, i) for i, ex in enumerate(examples)]
    prompts = [
        render_prompt(
            ex,
            tokenizer=answer_tokenizer,
            question_max_chars=prompt_question_max_chars,
            reference_max_chars=prompt_reference_max_chars,
        )
        for ex in normalized
    ]
    results = []
    prompt_token_lengths_raw = [len(answer_tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    orig_side = answer_tokenizer.padding_side
    orig_trunc_side = getattr(answer_tokenizer, "truncation_side", "right")
    answer_tokenizer.padding_side = "left"
    answer_tokenizer.truncation_side = "left"
    if answer_tokenizer.pad_token_id is None:
        answer_tokenizer.pad_token = answer_tokenizer.eos_token
    was_training = answer_model.training
    answer_model.eval()
    for start in range(0, len(prompts), size):
        prompts_part = prompts[start:start + size]
        examples_part = normalized[start:start + size]
        enc = answer_tokenizer(
            prompts_part,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        ).to(answer_model.device)
        out = answer_model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=answer_tokenizer.pad_token_id,
            eos_token_id=answer_tokenizer.eos_token_id,
        )
        input_len = enc["input_ids"].shape[1]
        for i, ex in enumerate(examples_part):
            gen_ids = out[i][input_len:]
            prediction = answer_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            used_len = int(enc["attention_mask"][i].sum().item())
            results.append({
                "id": ex["id"],
                "eval_uid": ex["eval_uid"],
                "raw_index": ex["raw_index"],
                "question": ex["question"],
                "reference": ex["reference"],
                "answer": ex["answer"],
                "prediction": prediction,
                "prompt_text": prompts_part[i],
                "debug_prompt_tokens_raw": prompt_token_lengths_raw[start + i],
                "debug_prompt_tokens_used": used_len,
                "debug_prompt_truncated": prompt_token_lengths_raw[start + i] > used_len,
            })
    answer_tokenizer.padding_side = orig_side
    answer_tokenizer.truncation_side = orig_trunc_side
    if was_training:
        answer_model.train()
    return results


def evaluate_policy(*, eval_data: List[Dict[str, Any]], adapted_model, adapted_tokenizer, base_model, base_tokenizer, judge_model, judge_tokenizer, max_length: int, max_new_tokens: int, judge_max_length: int, judge_max_new_tokens: int, save_dir: str, max_samples: int, prompt_question_max_chars: int, prompt_reference_max_chars: int, eval_size: int, debug_max_text_chars: int, eval_step: int):
    os.makedirs(save_dir, exist_ok=True)
    eval_subset = eval_data if (max_samples is None or max_samples < 0) else eval_data[:max_samples]
    subset = [normalize_example(ex, i) for i, ex in enumerate(eval_subset)]
    t0 = time.time()

    base_predictions_path = os.path.join(save_dir, "base_predictions.jsonl")
    adapted_predictions_path = os.path.join(save_dir, "adapted_predictions.jsonl")
    base_judge_path = os.path.join(save_dir, "base_judge.jsonl")
    adapted_judge_path = os.path.join(save_dir, "adapted_judge.jsonl")
    per_sample_diff_path = os.path.join(save_dir, "per_sample_diff.jsonl")
    chunk_dir = os.path.join(save_dir, "_eval_chunks")

    for path_ in (base_predictions_path, adapted_predictions_path, base_judge_path, adapted_judge_path, per_sample_diff_path):
        if os.path.exists(path_):
            os.remove(path_)
    os.makedirs(chunk_dir, exist_ok=True)
    for name in os.listdir(chunk_dir):
        try:
            os.remove(os.path.join(chunk_dir, name))
        except Exception:
            pass

    chunk_size = 1000
    if chunk_size <= 0:
        chunk_size = max(len(subset), 1)

    base_success_count = 0
    base_hallucinated_count = 0
    adapted_success_count = 0
    adapted_hallucinated_count = 0
    base_failure_count = 0
    adapted_failure_count = 0
    n_same = 0
    n_diff = 0
    n_improved = 0
    n_worsened = 0
    per_sample_diff_count = 0
    chunk_progress = []

    def write_chunk_then_append(rows: List[Dict[str, Any]], final_path: str, chunk_filename: str):
        if not rows:
            return
        chunk_path = os.path.join(chunk_dir, chunk_filename)
        write_jsonl(chunk_path, rows)
        append_jsonl(final_path, rows)
        try:
            os.remove(chunk_path)
        except Exception:
            pass

    def merge_same_prediction_base_result(pred_row: Dict[str, Any], base_judge_row: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(base_judge_row)
        row["id"] = str(pred_row.get("id", ""))
        row["eval_uid"] = str(pred_row.get("eval_uid", ""))
        return row

    for chunk_idx, start in enumerate(range(0, len(subset), chunk_size)):
        chunk = subset[start:start + chunk_size]
        base_key = prediction_cache_key(
            chunk,
            model_name_or_path=getattr(base_model, "name_or_path", getattr(base_model.config, "_name_or_path", "base")),
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            prompt_question_max_chars=prompt_question_max_chars,
            prompt_reference_max_chars=prompt_reference_max_chars,
        )
        if base_key in BASE_PRED_CACHE:
            base_predicted = BASE_PRED_CACHE[base_key]
        else:
            base_predicted = generate_predictions(
                chunk,
                answer_model=base_model,
                answer_tokenizer=base_tokenizer,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                prompt_question_max_chars=prompt_question_max_chars,
                prompt_reference_max_chars=prompt_reference_max_chars,
                size=eval_size,
            )
            BASE_PRED_CACHE[base_key] = base_predicted

        adapted_predicted = generate_predictions(
            chunk,
            answer_model=adapted_model,
            answer_tokenizer=adapted_tokenizer,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            prompt_question_max_chars=prompt_question_max_chars,
            prompt_reference_max_chars=prompt_reference_max_chars,
            size=eval_size,
        )

        base_align = check_id_alignment(chunk, base_predicted)
        adapted_align = check_id_alignment(chunk, adapted_predicted)
        if not base_align["aligned"]:
            raise RuntimeError(f"base eval alignment mismatch: {base_align}")
        if not adapted_align["aligned"]:
            raise RuntimeError(f"adapted eval alignment mismatch: {adapted_align}")

        base_succ, base_fail = judge_examples_model(
            base_predicted,
            judge_model=judge_model,
            judge_tokenizer=judge_tokenizer,
            judge_max_length=judge_max_length,
            judge_max_new_tokens=judge_max_new_tokens,
            size=eval_size,
            debug_max_text_chars=debug_max_text_chars,
        )
        base_judge_by_uid = {r["eval_uid"]: r for r in (base_succ + base_fail)}

        adapted_same_results: List[Dict[str, Any]] = []
        adapted_to_judge: List[Dict[str, Any]] = []
        for bp, ap in zip(base_predicted, adapted_predicted):
            if bp.get("prediction", "") == ap.get("prediction", ""):
                base_row = base_judge_by_uid.get(bp["eval_uid"], {"status": "missing_base_judge"})
                adapted_same_results.append(merge_same_prediction_base_result(ap, base_row))
            else:
                adapted_to_judge.append(ap)

        adapted_diff_succ, adapted_diff_fail = judge_examples_model(
            adapted_to_judge,
            judge_model=judge_model,
            judge_tokenizer=judge_tokenizer,
            judge_max_length=judge_max_length,
            judge_max_new_tokens=judge_max_new_tokens,
            size=eval_size,
            debug_max_text_chars=debug_max_text_chars,
        ) if adapted_to_judge else ([], [])

        adapted_judge_by_uid = {r["eval_uid"]: r for r in adapted_same_results + adapted_diff_succ + adapted_diff_fail}
        adapted_all = []
        for ap in adapted_predicted:
            row = adapted_judge_by_uid.get(ap["eval_uid"], {"id": ap["id"], "eval_uid": ap["eval_uid"], "status": "missing_adapted_judge"})
            adapted_all.append(row)

        base_succ_count_chunk = sum(1 for r in (base_succ + base_fail) if r.get("status") is None)
        adapted_succ_count_chunk = sum(1 for r in adapted_all if r.get("status") is None)
        base_hallucinated_count_chunk = sum(1 for r in base_succ if r.get("label") == "HALLUCINATED")
        adapted_hallucinated_count_chunk = sum(1 for r in adapted_all if r.get("status") is None and r.get("label") == "HALLUCINATED")
        base_failure_count_chunk = len(base_predicted) - base_succ_count_chunk
        adapted_failure_count_chunk = len(adapted_predicted) - adapted_succ_count_chunk

        base_success_count += base_succ_count_chunk
        adapted_success_count += adapted_succ_count_chunk
        base_hallucinated_count += base_hallucinated_count_chunk
        adapted_hallucinated_count += adapted_hallucinated_count_chunk
        base_failure_count += base_failure_count_chunk
        adapted_failure_count += adapted_failure_count_chunk

        per_sample_diff_chunk = []
        for bp, ap in zip(base_predicted, adapted_predicted):
            same_output = bp.get("prediction", "") == ap.get("prediction", "")
            if same_output:
                n_same += 1
            else:
                n_diff += 1
            uid = bp["eval_uid"]
            base_label = base_judge_by_uid.get(uid, {}).get("label", "UNKNOWN")
            adapted_label = adapted_judge_by_uid.get(uid, {}).get("label", "UNKNOWN")
            improved = base_label == "HALLUCINATED" and adapted_label == "NOT_HALLUCINATED"
            worsened = base_label == "NOT_HALLUCINATED" and adapted_label == "HALLUCINATED"
            n_improved += int(improved)
            n_worsened += int(worsened)
            if not same_output:
                per_sample_diff_chunk.append({
                    "id": bp["id"],
                    "eval_uid": uid,
                    "question": limit_text(bp.get("question", ""), debug_max_text_chars),
                    "reference": limit_text(bp.get("reference", ""), debug_max_text_chars),
                    "gold": limit_text(bp.get("answer", ""), debug_max_text_chars),
                    "base_pred": limit_text(bp.get("prediction", ""), debug_max_text_chars),
                    "adapted_pred": limit_text(ap.get("prediction", ""), debug_max_text_chars),
                    "base_label": base_label,
                    "adapted_label": adapted_label,
                    "improved": improved,
                    "worsened": worsened,
                })
        per_sample_diff_count += len(per_sample_diff_chunk)

        write_chunk_then_append(base_predicted, base_predictions_path, f"chunk_{chunk_idx:05d}_base_predictions.jsonl")
        write_chunk_then_append(adapted_predicted, adapted_predictions_path, f"chunk_{chunk_idx:05d}_adapted_predictions.jsonl")
        write_chunk_then_append(base_succ + base_fail, base_judge_path, f"chunk_{chunk_idx:05d}_base_judge.jsonl")
        write_chunk_then_append(adapted_all, adapted_judge_path, f"chunk_{chunk_idx:05d}_adapted_judge.jsonl")
        write_chunk_then_append(per_sample_diff_chunk, per_sample_diff_path, f"chunk_{chunk_idx:05d}_per_sample_diff.jsonl")

        chunk_progress.append({
            "chunk_idx": int(chunk_idx),
            "start": int(start),
            "end_exclusive": int(start + len(chunk)),
            "chunk_size": int(len(chunk)),
            "base_successes": int(base_succ_count_chunk),
            "adapted_successes": int(adapted_succ_count_chunk),
            "base_hallucinated": int(base_hallucinated_count_chunk),
            "adapted_hallucinated": int(adapted_hallucinated_count_chunk),
            "same_output": int(sum(1 for bp, ap in zip(base_predicted, adapted_predicted) if bp.get("prediction", "") == ap.get("prediction", ""))),
            "different_output": int(sum(1 for bp, ap in zip(base_predicted, adapted_predicted) if bp.get("prediction", "") != ap.get("prediction", ""))),
            "per_sample_diff_rows": int(len(per_sample_diff_chunk)),
        })

    bm = {
        "num_samples": int(base_success_count),
        "num_hallucinated": int(base_hallucinated_count),
        "hallucination_rate": float(base_hallucinated_count / max(base_success_count, 1)),
        "num_correct": int(base_success_count - base_hallucinated_count),
        "num_failures": int(base_failure_count),
    }
    am = {
        "num_samples": int(adapted_success_count),
        "num_hallucinated": int(adapted_hallucinated_count),
        "hallucination_rate": float(adapted_hallucinated_count / max(adapted_success_count, 1)),
        "num_correct": int(adapted_success_count - adapted_hallucinated_count),
        "num_failures": int(adapted_failure_count),
    }

    comparison = {
        "num_samples": len(subset),
        "step": int(eval_step),
        "eval_chunk_size": int(chunk_size),
        "num_chunks": int(len(chunk_progress)),
        "base": bm,
        "adapted": am,
        "delta": {
            "hallucination_rate": round(am["hallucination_rate"] - bm["hallucination_rate"], 6),
            "num_hallucinated": am["num_hallucinated"] - bm["num_hallucinated"],
            "num_correct": am["num_correct"] - bm["num_correct"],
        },
        "output_diff": {"same": int(n_same), "different": int(n_diff)},
        "label_diff": {"improved": int(n_improved), "worsened": int(n_worsened)},
        "per_sample_diff_count": int(per_sample_diff_count),
        "per_sample_diff_path": os.path.basename(per_sample_diff_path),
        "chunk_progress_path": "chunk_progress.json",
        "chunk_progress": chunk_progress,
        "eval_time_sec": round(time.time() - t0, 1),
    }

    write_json(os.path.join(save_dir, "chunk_progress.json"), {"chunks": chunk_progress})
    write_json(os.path.join(save_dir, "comparison.json"), comparison)

    try:
        os.rmdir(chunk_dir)
    except Exception:
        pass
    return comparison

