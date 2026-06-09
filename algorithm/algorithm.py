from typing import Any, Dict, List, Sequence, Tuple

import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.util import answers_match, normalize_text, try_parse_number


class FormulaHallucinationObjective(nn.Module):
    def __init__(self, args, model, tokenizer, base_model=None, base_tokenizer=None):
        super().__init__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer if base_tokenizer is not None else tokenizer
        self.hidden_size = int(model.config.hidden_size)
        self.semantic_match_cache: Dict[str, Dict[str, Any]] = {}

        self.layers = self.get_layers(model)
        self.layer_idx = self.normalize_layer_idx(args.trainable_target_layer_idx, len(self.layers))
        self.target_layer = self.layers[self.layer_idx]

        self.row_mask_summary: List[Dict[str, Any]] = []
        self.freeze_all_model_params()
        self.activate_partial_target_layer()

    def get_layers(self, model):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        raise ValueError("Unsupported model structure: cannot find transformer layers")

    def normalize_layer_idx(self, idx, num_layers):
        if idx < 0:
            idx = num_layers + idx
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Invalid trainable_target_layer_idx={idx}, num_layers={num_layers}")
        return idx

    def freeze_all_model_params(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def make_hidden_mask(self, size, ratio, select_mode, seed):
        ratio = max(0.0, min(1.0, ratio))
        k = max(1, int(round(size * ratio)))
        k = min(k, size)
        mask = torch.zeros(size, dtype=torch.float32)
        if select_mode == "first":
            idx = torch.arange(k)
        else:
            g = torch.Generator()
            g.manual_seed(seed)
            idx = torch.randperm(size, generator=g)[:k]
        mask[idx] = 1.0
        return mask

    def register_row_mask_hook(self, param, row_mask):
        mask = row_mask[:, None] if param.ndim == 2 else row_mask

        def hook(grad):
            return grad * mask.to(device=grad.device, dtype=grad.dtype)

        param.register_hook(hook)

    def activate_partial_target_layer(self):
        row_mask = self.make_hidden_mask(
            size=self.hidden_size,
            ratio=self.args.trainable_ratio,
            select_mode=self.args.trainable_select,
            seed=self.args.trainable_seed,
        )
        self.register_buffer("row_mask", row_mask)
        for module_name, module in self.target_layer.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if module.weight.shape[0] != self.hidden_size:
                continue
            module.weight.requires_grad = True
            self.register_row_mask_hook(module.weight, self.row_mask)
            self.row_mask_summary.append({
                "module_name": module_name,
                "out_dim": int(module.weight.shape[0]),
                "active_rows": int(self.row_mask.sum().item()),
            })
            if module.bias is not None:
                module.bias.requires_grad = True
                self.register_row_mask_hook(module.bias, self.row_mask)

    def describe(self):
        lines = [
            f"trainable_target_layer_idx={self.layer_idx}",
            f"hidden_size={self.hidden_size}",
            f"trainable_rows={int(self.row_mask.sum().item())}/{self.hidden_size}",
            f"train_pipeline={getattr(self.args, 'train_pipeline', 'new_method')}",
            f"formula=conditional_norm(CE, CONF) with base-generated bad answers + divergence-to-end probability margin | match_mode={self.args.match_mode}",
        ]
        for row in self.row_mask_summary:
            lines.append(f"{row['module_name']}: active_rows={row['active_rows']}/{row['out_dim']}")
        return "\n".join(lines)

    def answer_only_system_prompt(self, has_reference: bool):
        if has_reference:
            return (
                "You are a precise question-answering assistant. "
                "Answer the user's question using only the provided reference. "
                "Do not explain your reasoning. Do not restate the question. "
                "Return only the final answer text. If the answer is a number, return only the number."
            )
        return (
            "You are a precise question-answering assistant. "
            "Answer the user's question as accurately as possible. "
            "Do not explain your reasoning. Do not restate the question. "
            "Return only the final answer text. If the answer is a number, return only the number."
        )

    def build_prompt(self, question, reference):
        question = "" if question is None else str(question)
        reference = "" if reference is None else str(reference)
        if self.args.prompt_question_max_chars > 0:
            question = question[: self.args.prompt_question_max_chars]
        if self.args.prompt_reference_max_chars > 0:
            reference = reference[: self.args.prompt_reference_max_chars]
        has_reference = bool(reference.strip())
        system = self.answer_only_system_prompt(has_reference)
        if has_reference:
            user = f"REFERENCE:\n{reference}\n\nQUESTION:\n{question}\n\nReturn only the final answer."
        else:
            user = f"QUESTION:\n{question}\n\nReturn only the final answer."
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"

    def build_answer_text(self, answer):
        answer = "" if answer is None else str(answer).strip()
        if self.tokenizer.eos_token is None:
            return answer
        return self.tokenizer.eos_token if answer == "" else answer + self.tokenizer.eos_token

    def encode_prompt_only(self, prompt_text):
        prompt_ids_raw = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_ids = list(prompt_ids_raw)
        prompt_raw_len = len(prompt_ids)
        prompt_truncated = False
        if len(prompt_ids) > self.args.max_length:
            prompt_ids = prompt_ids[-self.args.max_length :]
            prompt_truncated = True
        return {
            "prompt_ids": prompt_ids,
            "prompt_attention_mask": [1] * len(prompt_ids),
            "prompt_raw_len": prompt_raw_len,
            "prompt_used_len": len(prompt_ids),
            "prompt_truncated": prompt_truncated,
        }

    def encode_prompt_answer(self, prompt_text, answer_text):
        prompt_ids_raw = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids_raw = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        prompt_ids = list(prompt_ids_raw)
        answer_ids = list(answer_ids_raw)
        prompt_raw_len = len(prompt_ids)
        answer_raw_len = len(answer_ids)
        answer_truncated = False
        if len(answer_ids) > self.args.max_length:
            answer_ids = answer_ids[: self.args.max_length]
            answer_truncated = True
            if self.tokenizer.eos_token_id is not None and len(answer_ids) > 0:
                answer_ids[-1] = self.tokenizer.eos_token_id
        max_prompt_len = max(0, self.args.max_length - len(answer_ids))
        prompt_truncated = len(prompt_ids) > max_prompt_len
        if prompt_truncated:
            prompt_ids = prompt_ids[-max_prompt_len:]
        input_ids = prompt_ids + answer_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + answer_ids
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_len": len(prompt_ids),
            "answer_ids": answer_ids,
            "prompt_raw_len": prompt_raw_len,
            "answer_raw_len": answer_raw_len,
            "prompt_used_len": len(prompt_ids),
            "answer_used_len": len(answer_ids),
            "prompt_truncated": prompt_truncated,
            "answer_truncated": answer_truncated,
        }

    def left_pad(self, sequences, pad_value):
        max_len = max(len(x) for x in sequences)
        rows = []
        for x in sequences:
            pad_len = max_len - len(x)
            rows.append([pad_value] * pad_len + x)
        return torch.tensor(rows, dtype=torch.long)

    def collate(self, examples):
        ids, questions, references, answers, prompt_texts = [], [], [], [], []
        gold_rows, prompt_only_rows = [], []
        answer_token_ids = []
        for ex in examples:
            prompt_text = self.build_prompt(ex["question"], ex["reference"])
            answer_text = self.build_answer_text(ex["answer"])
            gold = self.encode_prompt_answer(prompt_text, answer_text)
            prompt_only = self.encode_prompt_only(prompt_text)
            ids.append(ex["id"])
            questions.append(ex["question"])
            references.append(ex["reference"])
            answers.append(ex["answer"])
            prompt_texts.append(prompt_text)
            gold_rows.append(gold)
            prompt_only_rows.append(prompt_only)
            answer_token_ids.append(self.tokenizer(answer_text, add_special_tokens=False)["input_ids"])
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        return {
            "id": ids,
            "question": questions,
            "reference": references,
            "answer": answers,
            "prompt_text": prompt_texts,
            "gold_answer_token_ids": answer_token_ids,
            "prompt_input_ids": self.left_pad([x["prompt_ids"] for x in prompt_only_rows], pad_id),
            "prompt_attention_mask": self.left_pad([x["prompt_attention_mask"] for x in prompt_only_rows], 0),
            "prompt_raw_len": [x["prompt_raw_len"] for x in prompt_only_rows],
            "prompt_used_len": [x["prompt_used_len"] for x in prompt_only_rows],
            "prompt_truncated": [x["prompt_truncated"] for x in prompt_only_rows],
            "gold_input_ids": self.left_pad([x["input_ids"] for x in gold_rows], pad_id),
            "gold_attention_mask": self.left_pad([x["attention_mask"] for x in gold_rows], 0),
            "gold_labels": self.left_pad([x["labels"] for x in gold_rows], -100),
            "gold_prompt_len": [x["prompt_len"] for x in gold_rows],
            "gold_prompt_raw_len": [x["prompt_raw_len"] for x in gold_rows],
            "gold_answer_raw_len": [x["answer_raw_len"] for x in gold_rows],
            "gold_prompt_used_len": [x["prompt_used_len"] for x in gold_rows],
            "gold_answer_used_len": [x["answer_used_len"] for x in gold_rows],
            "gold_prompt_truncated": [x["prompt_truncated"] for x in gold_rows],
            "gold_answer_truncated": [x["answer_truncated"] for x in gold_rows],
        }

    def move_to_device(self, batch: Dict[str, Any]):
        device = next(self.model.parameters()).device
        out = {}
        for k, v in batch.items():
            out[k] = v.to(device) if torch.is_tensor(v) else v
        return out

    @torch.no_grad()
    def _generate_answers_with_model(self, model, decode_tokenizer, prompt_input_ids, prompt_attention_mask):
        was_training = model.training
        model.eval()
        old_use_cache = getattr(model.config, "use_cache", True)
        model.config.use_cache = True
        max_new = self.args.method_self_max_new_tokens if self.args.method_self_max_new_tokens > 0 else self.args.max_new_tokens
        out = model.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_new,
            do_sample=False,
            use_cache=True,
            pad_token_id=decode_tokenizer.pad_token_id,
            eos_token_id=decode_tokenizer.eos_token_id,
        )
        input_len = prompt_input_ids.shape[1]
        answers = []
        token_ids = []
        for row in out:
            gen_ids = row[input_len:]
            answers.append(decode_tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
            answer_text = self.build_answer_text(answers[-1])
            token_ids.append(self.tokenizer(answer_text, add_special_tokens=False)["input_ids"])
        model.config.use_cache = old_use_cache
        if was_training:
            model.train()
        return answers, token_ids

    @torch.no_grad()
    def generate_base_answers(self, batch: Dict[str, Any]) -> Tuple[List[str], List[List[int]]]:
        if self.base_model is None:
            raise ValueError("Base model is required to generate base answers.")
        prompt_input_ids = batch["prompt_input_ids"]
        prompt_attention_mask = batch["prompt_attention_mask"]
        device = next(self.base_model.parameters()).device
        return self._generate_answers_with_model(
            self.base_model,
            self.base_tokenizer,
            prompt_input_ids.to(device),
            prompt_attention_mask.to(device),
        )

    def build_answer_batch(self, batch, answers: Sequence[str], prefix: str):
        rows = []
        for prompt_text, answer in zip(batch["prompt_text"], answers):
            rows.append(self.encode_prompt_answer(prompt_text, self.build_answer_text(answer)))
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        device = batch["gold_input_ids"].device
        return {
            f"{prefix}_input_ids": self.left_pad([x["input_ids"] for x in rows], pad_id).to(device),
            f"{prefix}_attention_mask": self.left_pad([x["attention_mask"] for x in rows], 0).to(device),
            f"{prefix}_labels": self.left_pad([x["labels"] for x in rows], -100).to(device),
            f"{prefix}_prompt_len": [x["prompt_len"] for x in rows],
        }

    def compute_token_ce(self, logits, labels):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        vocab = shift_logits.shape[-1]
        token_loss = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)
        mask = shift_labels != -100
        seq_means = []
        token_lists = []
        for i in range(shift_labels.shape[0]):
            vals = token_loss[i][mask[i]]
            token_lists.append(vals)
            seq_means.append(vals.mean() if vals.numel() > 0 else token_loss.new_tensor(0.0))
        return token_lists, torch.stack(seq_means)

    def first_divergence_index(self, gold_ids: Sequence[int], bad_ids: Sequence[int]) -> int:
        n = min(len(gold_ids), len(bad_ids))
        for i in range(n):
            if int(gold_ids[i]) != int(bad_ids[i]):
                return i
        if len(gold_ids) != len(bad_ids):
            return n
        return 0

    def slice_from_divergence(self, values: torch.Tensor, start: int):
        if values.numel() == 0:
            return values.new_zeros((0,))
        start = max(0, min(int(start), max(values.shape[0] - 1, 0)))
        return values[start:]

    def safe_mean(self, values: torch.Tensor):
        return values.mean() if values.numel() > 0 else values.new_tensor(0.0)

    def _match_cache_key(self, question: Any, gold: Any, pred: Any) -> str:
        return " || ".join([normalize_text(question), normalize_text(gold), normalize_text(pred)])

    def _render_semantic_match_prompt(self, question: Any, gold: Any, pred: Any) -> str:
        question = "" if question is None else str(question).strip()
        gold = "" if gold is None else str(gold).strip()
        pred = "" if pred is None else str(pred).strip()
        system = (
            "Judge whether GOLD_ANSWER and CANDIDATE_ANSWER mean the same final answer to QUESTION. "
            "Ignore wording differences, abbreviations, aliases, extra names, and formatting if they refer to the same answer. "
            "Output only SAME or DIFFERENT."
        )
        user = (
            f"QUESTION:\n{question}\n\n"
            f"GOLD_ANSWER:\n{gold}\n\n"
            f"CANDIDATE_ANSWER:\n{pred}\n\n"
            "LABEL:"
        )
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"

    def _parse_semantic_match_label(self, text: Any) -> str | None:
        text = "" if text is None else str(text)
        m = re.search(r"\b(SAME|DIFFERENT)\b", text, flags=re.IGNORECASE)
        return None if m is None else m.group(1).upper()

    def _rule_match_record(self, question: Any, gold: Any, pred: Any, *, status: str) -> Dict[str, Any]:
        same = answers_match(gold, pred, mode="numeric_or_exact")
        return {
            "match_mode": str(self.args.match_mode),
            "match_same": bool(same),
            "match_label": "SAME" if same else "DIFFERENT",
            "match_status": status,
            "match_raw_output": "",
            "match_prompt_text": "",
        }

    @torch.no_grad()
    def semantic_match_answers(self, questions: Sequence[Any], gold_answers: Sequence[Any], pred_answers: Sequence[Any]) -> List[Dict[str, Any]]:
        if str(getattr(self.args, "match_mode", "numeric_or_exact")) != "llm_semantic":
            return [
                self._rule_match_record(q, g, p, status="rule_fallback")
                if str(getattr(self.args, "match_mode", "numeric_or_exact")) != "exact"
                else {
                    "match_mode": str(self.args.match_mode),
                    "match_same": bool(answers_match(g, p, mode="exact")),
                    "match_label": "SAME" if answers_match(g, p, mode="exact") else "DIFFERENT",
                    "match_status": "rule_exact",
                    "match_raw_output": "",
                    "match_prompt_text": "",
                }
                for q, g, p in zip(questions, gold_answers, pred_answers)
            ]

        out: List[Dict[str, Any] | None] = [None for _ in range(len(gold_answers))]
        to_run = []
        for i, (q, g, p) in enumerate(zip(questions, gold_answers, pred_answers)):
            g_norm = normalize_text(g)
            p_norm = normalize_text(p)
            gn = try_parse_number(g)
            pn = try_parse_number(p)
            if g_norm == p_norm:
                out[i] = {
                    "match_mode": "llm_semantic",
                    "match_same": True,
                    "match_label": "SAME",
                    "match_status": "rule_exact_equal",
                    "match_raw_output": "",
                    "match_prompt_text": "",
                }
                continue
            if gn is not None and pn is not None and abs(gn - pn) < 1e-9:
                out[i] = {
                    "match_mode": "llm_semantic",
                    "match_same": True,
                    "match_label": "SAME",
                    "match_status": "rule_numeric_equal",
                    "match_raw_output": "",
                    "match_prompt_text": "",
                }
                continue
            key = self._match_cache_key(q, g, p)
            cached = self.semantic_match_cache.get(key)
            if cached is not None:
                out[i] = dict(cached)
                out[i]["match_status"] = "cache_" + str(out[i].get("match_status", "ok"))
                continue
            prompt = self._render_semantic_match_prompt(q, g, p)
            to_run.append((i, key, prompt))

        if to_run:
            model = self.model
            tokenizer = self.tokenizer
            orig_training = model.training
            orig_use_cache = getattr(model.config, "use_cache", True)
            orig_padding_side = tokenizer.padding_side
            orig_trunc_side = getattr(tokenizer, "truncation_side", "right")
            model.eval()
            model.config.use_cache = True
            tokenizer.padding_side = "left"
            tokenizer.truncation_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            batch_size = max(1, int(getattr(self.args, "match_judge_batch_size", 8)))
            max_len = int(getattr(self.args, "match_judge_max_length", 1024))
            max_new = int(getattr(self.args, "match_judge_max_new_tokens", 4))
            try:
                for start in range(0, len(to_run), batch_size):
                    part = to_run[start:start + batch_size]
                    prompts = [x[2] for x in part]
                    enc = tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_len,
                        add_special_tokens=False,
                    ).to(next(model.parameters()).device)
                    gen = model.generate(
                        **enc,
                        max_new_tokens=max_new,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    input_len = enc["input_ids"].shape[1]
                    for row_idx, (sample_idx, key, prompt_text) in enumerate(part):
                        gen_ids = gen[row_idx][input_len:]
                        raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                        label = self._parse_semantic_match_label(raw_output)
                        if label is None:
                            same = answers_match(gold_answers[sample_idx], pred_answers[sample_idx], mode="numeric_or_exact")
                            record = {
                                "match_mode": "llm_semantic",
                                "match_same": bool(same),
                                "match_label": "SAME" if same else "DIFFERENT",
                                "match_status": "parse_fallback",
                                "match_raw_output": raw_output,
                                "match_prompt_text": prompt_text,
                            }
                        else:
                            record = {
                                "match_mode": "llm_semantic",
                                "match_same": bool(label == "SAME"),
                                "match_label": label,
                                "match_status": "llm_ok",
                                "match_raw_output": raw_output,
                                "match_prompt_text": prompt_text,
                            }
                        self.semantic_match_cache[key] = dict(record)
                        out[sample_idx] = record
            finally:
                tokenizer.padding_side = orig_padding_side
                tokenizer.truncation_side = orig_trunc_side
                model.config.use_cache = orig_use_cache
                if orig_training:
                    model.train()

        return [x if x is not None else self._rule_match_record(q, g, p, status="final_fallback") for x, q, g, p in zip(out, questions, gold_answers, pred_answers)]

    def activation(self, x: torch.Tensor):
        kind = str(getattr(self.args, "method_activation", "relu")).lower().strip()
        beta = float(getattr(self.args, "method_activation_beta", 1.0))
        if kind == "relu":
            return F.relu(x)
        if kind == "sigmoid":
            return torch.sigmoid(beta * x)
        return F.softplus(beta * x) / max(beta, 1e-8)

    def bounded_gate(self, deficit: torch.Tensor):
        act = self.activation(deficit)
        c = max(float(getattr(self.args, "method_gate_scale", 1.0)), 1e-8)
        return act / (act + act.new_tensor(c))

    def _scalar(self, value: torch.Tensor | float | int) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().item())
        return float(value)

    def _vector(self, value: torch.Tensor) -> List[float]:
        if value.numel() == 0:
            return []
        return [float(v) for v in value.detach().float().cpu().tolist()]

    def forward_sft_only(self, batch: Dict[str, Any]):
        batch = self.move_to_device(batch)
        batch_size = len(batch["answer"])

        gold_outputs = self.model(
            input_ids=batch["gold_input_ids"],
            attention_mask=batch["gold_attention_mask"],
            labels=batch["gold_labels"],
            use_cache=False,
            return_dict=True,
        )
        gold_token_ce, gold_seq_ce = self.compute_token_ce(gold_outputs.logits, batch["gold_labels"])
        del gold_outputs

        ce_num = gold_seq_ce.sum()
        ce_denom = gold_seq_ce.new_tensor(float(batch_size))
        ce_base_loss = ce_num / ce_denom.clamp_min(1e-8) if batch_size > 0 else gold_seq_ce.new_tensor(0.0)
        total_loss = ce_base_loss

        sample_metrics = []
        for i in range(batch_size):
            sample_metrics.append({
                "sample_index": int(i),
                "base_mismatch": False,
                "match_mode": "sft_only",
                "match_same": True,
                "match_label": "",
                "match_status": "sft_only",
                "match_raw_output": "",
                "match_prompt_text": "",
                "need_bad_branch": False,
                "divergence_index": 0,
                "gold_seq_ce": self._scalar(gold_seq_ce[i]),
                "bad_seq_ce": 0.0,
                "gold_token_ce": self._vector(gold_token_ce[i]),
                "bad_token_ce": [],
                "gold_divergence_ce": [],
                "bad_divergence_ce": [],
                "gold_divergence_ce_mean": 0.0,
                "bad_divergence_ce_mean": 0.0,
                "prob_margin": 0.0,
                "prob_margin_target": 0.0,
                "halluci_margin_threshold": 0.0,
                "bad_ce_target": 0.0,
                "deficit": 0.0,
                "gate": 0.0,
                "bad_term": 0.0,
                "conf": 0.0,
                "active_halluci": False,
                "selected": True,
                "ce_weight": 1.0,
                "sample_ce_num": self._scalar(gold_seq_ce[i]),
                "sample_conf_num": 0.0,
                "used_ce": True,
                "used_conf": False,
                "used_union": True,
            })

        zero = total_loss.new_tensor(0.0)
        return {
            "loss": total_loss,
            "ce_base_loss": ce_base_loss,
            "conf_loss": zero,
            "gold_seq_ce_mean": gold_seq_ce.mean() if gold_seq_ce.numel() > 0 else zero,
            "bad_seq_ce_mean": zero,
            "raw_mag_mean": zero,
            "mag_mean": zero,
            "mag_pos_mean": zero,
            "mag_neg_mean": zero,
            "divergence_ce_gold_mean": zero,
            "divergence_ce_bad_mean": zero,
            "rank_loss_mean": zero,
            "ce_contrib": ce_base_loss,
            "conf_contrib": zero,
            "scale_ce": 1.0,
            "scale_conf": 0.0,
            "num_halluci": 0,
            "num_clean": int(batch_size),
            "num_selected": int(batch_size),
            "num_used_ce": int(batch_size),
            "num_used_conf": 0,
            "num_used_union": int(batch_size),
            "num_conf_eligible": 0,
            "num_base_mismatch": 0,
            "num_total_samples": int(batch_size),
            "batch_size": int(batch_size),
            "halluci_mask": [False] * batch_size,
            "base_mismatch_mask": [False] * batch_size,
            "divergence_indices": [0] * batch_size,
            "base_answers": [""] * batch_size,
            "match_records": [],
            "ce_weights": [1.0] * batch_size,
            "has_train_signal": bool(batch_size > 0),
            "ce_num": ce_num,
            "ce_denom": ce_denom,
            "conf_num": zero,
            "conf_denom": zero,
            "sample_ce_num": gold_seq_ce.detach(),
            "sample_conf_num": gold_seq_ce.new_zeros((batch_size,)),
            "sample_metrics": sample_metrics,
        }

    def forward(
        self,
        batch: Dict[str, Any],
        base_answers: Sequence[str] | None = None,
        base_answer_token_ids: Sequence[Sequence[int]] | None = None,
        balance_scales: Dict[str, float] | None = None,
        update_mag_stats: bool = True,
    ):
        del balance_scales
        del update_mag_stats
        batch = self.move_to_device(batch)

        if base_answers is None or base_answer_token_ids is None:
            base_answers, base_answer_token_ids = self.generate_base_answers(batch)

        batch_size = len(batch["answer"])
        match_records = self.semantic_match_answers(batch["question"], batch["answer"], base_answers)
        base_mismatch_mask = []
        divergence_indices = []
        for i in range(batch_size):
            base_mismatch = not bool(match_records[i].get("match_same", False))
            base_mismatch_mask.append(base_mismatch)
            divergence_indices.append(
                int(self.first_divergence_index(batch["gold_answer_token_ids"][i], base_answer_token_ids[i])) if base_mismatch else 0
            )

        need_bad_branch = any(base_mismatch_mask) and float(self.args.method_conf_coef) != 0.0

        gold_outputs = self.model(
            input_ids=batch["gold_input_ids"],
            attention_mask=batch["gold_attention_mask"],
            labels=batch["gold_labels"],
            use_cache=False,
            return_dict=True,
        )
        gold_token_ce, gold_seq_ce = self.compute_token_ce(gold_outputs.logits, batch["gold_labels"])
        del gold_outputs

        if need_bad_branch:
            bad_batch = self.build_answer_batch(batch, base_answers, prefix="bad")
            bad_outputs = self.model(
                input_ids=bad_batch["bad_input_ids"],
                attention_mask=bad_batch["bad_attention_mask"],
                labels=bad_batch["bad_labels"],
                use_cache=False,
                return_dict=True,
            )
            bad_token_ce, bad_seq_ce = self.compute_token_ce(bad_outputs.logits, bad_batch["bad_labels"])
            del bad_outputs
            del bad_batch
        else:
            bad_token_ce = [gold_seq_ce.new_zeros((0,)) for _ in range(batch_size)]
            bad_seq_ce = gold_seq_ce.new_zeros((batch_size,))

        raw_margin_vals = []
        act_vals = []
        margin_pos_vals = []
        margin_neg_vals = []
        divergence_ce_gold_vals = []
        divergence_ce_bad_vals = []
        deficit_vals = []
        conf_vals = []
        halluci_mask = []
        ce_weights = []
        conf_num_pieces = []
        ce_num_pieces = []
        ce_denom = gold_seq_ce.new_tensor(0.0)
        conf_denom = gold_seq_ce.new_tensor(0.0)

        prob_margin_target = float(getattr(self.args, "method_prob_margin_target", 0.0))
        bad_ce_target = float(getattr(self.args, "method_bad_ce_target", 2.0))
        halluci_margin_threshold = float(getattr(self.args, "method_halluci_margin_threshold", 0.0))

        sample_metrics = []
        sample_ce_num_vals = []
        sample_conf_num_vals = []
        used_ce_count = 0
        used_conf_count = 0
        used_union_count = 0

        for i in range(batch_size):
            zero = gold_seq_ce.new_tensor(0.0)
            base_mismatch = base_mismatch_mask[i]
            if base_mismatch and need_bad_branch:
                div_idx = divergence_indices[i]
                gold_divergence_ce = self.slice_from_divergence(gold_token_ce[i], div_idx)
                bad_divergence_ce = self.slice_from_divergence(bad_token_ce[i], div_idx)
                gold_divergence_ce_mean = self.safe_mean(gold_divergence_ce)
                bad_divergence_ce_mean = self.safe_mean(bad_divergence_ce)
                prob_margin = bad_divergence_ce_mean - gold_divergence_ce_mean
                active_halluci = bool(float(prob_margin.item()) < halluci_margin_threshold)
                deficit = F.relu(gold_seq_ce.new_tensor(prob_margin_target) - prob_margin)
                gate = self.bounded_gate(deficit) if active_halluci else zero
                bad_term = F.relu(gold_seq_ce.new_tensor(bad_ce_target) - bad_divergence_ce_mean) / max(bad_ce_target, 1e-8)
                bad_term = bad_term.clamp(min=0.0, max=1.0)
                conf = gate * bad_term if active_halluci else zero
            else:
                gold_divergence_ce = gold_seq_ce.new_zeros((0,))
                bad_divergence_ce = gold_seq_ce.new_zeros((0,))
                gold_divergence_ce_mean = zero
                bad_divergence_ce_mean = zero
                prob_margin = zero
                deficit = zero
                gate = zero
                bad_term = zero
                conf = zero
                active_halluci = False

            halluci_mask.append(active_halluci)
            selected = (active_halluci and self.args.train_use_halluci) or ((not active_halluci) and self.args.train_use_clean)
            ce_weight = (
                self.args.train_halluci_ce_weight if (active_halluci and selected)
                else (self.args.train_clean_ce_weight if selected else 0.0)
            )
            ce_weights.append(ce_weight)
            sample_ce_num = gold_seq_ce[i] * float(ce_weight) if ce_weight > 0.0 else zero
            sample_conf_num = conf if active_halluci else zero
            sample_ce_num_vals.append(sample_ce_num)
            sample_conf_num_vals.append(sample_conf_num)
            used_ce = bool(ce_weight > 0.0)
            used_conf = bool(active_halluci and float(self.args.method_conf_coef) != 0.0)
            used_union = bool(used_ce or used_conf)
            if used_ce:
                used_ce_count += 1
                ce_num_pieces.append(sample_ce_num)
                ce_denom = ce_denom + gold_seq_ce.new_tensor(float(ce_weight))
            if used_conf:
                used_conf_count += 1
            if used_union:
                used_union_count += 1

            raw_margin_vals.append(prob_margin)
            act_vals.append(gate)
            margin_pos_vals.append(F.relu(prob_margin))
            margin_neg_vals.append(F.relu(-prob_margin))
            divergence_ce_gold_vals.append(gold_divergence_ce_mean)
            divergence_ce_bad_vals.append(bad_divergence_ce_mean)
            deficit_vals.append(deficit)
            conf_vals.append(conf)

            if active_halluci:
                conf_num_pieces.append(conf)
                conf_denom = conf_denom + gold_seq_ce.new_tensor(1.0)

            sample_metrics.append({
                "sample_index": int(i),
                "base_mismatch": bool(base_mismatch),
                "match_mode": str(match_records[i].get("match_mode", self.args.match_mode)),
                "match_same": bool(match_records[i].get("match_same", not base_mismatch)),
                "match_label": str(match_records[i].get("match_label", "")),
                "match_status": str(match_records[i].get("match_status", "")),
                "match_raw_output": str(match_records[i].get("match_raw_output", "")),
                "match_prompt_text": str(match_records[i].get("match_prompt_text", "")),
                "need_bad_branch": bool(need_bad_branch),
                "divergence_index": int(divergence_indices[i]),
                "gold_seq_ce": self._scalar(gold_seq_ce[i]),
                "bad_seq_ce": self._scalar(bad_seq_ce[i]),
                "gold_token_ce": self._vector(gold_token_ce[i]),
                "bad_token_ce": self._vector(bad_token_ce[i]),
                "gold_divergence_ce": self._vector(gold_divergence_ce),
                "bad_divergence_ce": self._vector(bad_divergence_ce),
                "gold_divergence_ce_mean": self._scalar(gold_divergence_ce_mean),
                "bad_divergence_ce_mean": self._scalar(bad_divergence_ce_mean),
                "prob_margin": self._scalar(prob_margin),
                "prob_margin_target": float(prob_margin_target),
                "halluci_margin_threshold": float(halluci_margin_threshold),
                "bad_ce_target": float(bad_ce_target),
                "deficit": self._scalar(deficit),
                "gate": self._scalar(gate),
                "bad_term": self._scalar(bad_term),
                "conf": self._scalar(conf),
                "active_halluci": bool(active_halluci),
                "selected": bool(selected),
                "ce_weight": float(ce_weight),
                "sample_ce_num": self._scalar(sample_ce_num),
                "sample_conf_num": self._scalar(sample_conf_num),
                "used_ce": bool(used_ce),
                "used_conf": bool(used_conf),
                "used_union": bool(used_union),
            })

        raw_margin_t = torch.stack(raw_margin_vals) if raw_margin_vals else gold_seq_ce.new_zeros((0,))
        act_t = torch.stack(act_vals) if act_vals else gold_seq_ce.new_zeros((0,))
        margin_pos_t = torch.stack(margin_pos_vals) if margin_pos_vals else gold_seq_ce.new_zeros((0,))
        margin_neg_t = torch.stack(margin_neg_vals) if margin_neg_vals else gold_seq_ce.new_zeros((0,))
        divergence_ce_gold_t = torch.stack(divergence_ce_gold_vals) if divergence_ce_gold_vals else gold_seq_ce.new_zeros((0,))
        divergence_ce_bad_t = torch.stack(divergence_ce_bad_vals) if divergence_ce_bad_vals else gold_seq_ce.new_zeros((0,))
        deficit_t = torch.stack(deficit_vals) if deficit_vals else gold_seq_ce.new_zeros((0,))
        conf_t = torch.stack(conf_vals) if conf_vals else gold_seq_ce.new_zeros((0,))
        sample_ce_num_t = torch.stack(sample_ce_num_vals) if sample_ce_num_vals else gold_seq_ce.new_zeros((0,))
        sample_conf_num_t = torch.stack(sample_conf_num_vals) if sample_conf_num_vals else gold_seq_ce.new_zeros((0,))

        ce_num = torch.stack(ce_num_pieces).sum() if ce_num_pieces else gold_seq_ce.new_tensor(0.0)
        conf_num = torch.stack(conf_num_pieces).sum() if conf_num_pieces else gold_seq_ce.new_tensor(0.0)

        ce_base_loss = ce_num / ce_denom.clamp_min(1e-8) if float(ce_denom.item()) > 0.0 else gold_seq_ce.new_tensor(0.0)
        conf_loss = conf_num / conf_denom.clamp_min(1e-8) if float(conf_denom.item()) > 0.0 else gold_seq_ce.new_tensor(0.0)

        total_loss = (
            float(self.args.method_sft_coef) * ce_base_loss
            + float(self.args.method_conf_coef) * conf_loss
        )

        has_train_signal = bool(float(ce_denom.item()) > 0.0 or float(conf_denom.item()) > 0.0)
        return {
            "loss": total_loss,
            "ce_base_loss": ce_base_loss,
            "conf_loss": conf_loss,
            "gold_seq_ce_mean": gold_seq_ce.mean() if gold_seq_ce.numel() > 0 else total_loss.new_tensor(0.0),
            "bad_seq_ce_mean": bad_seq_ce.mean() if bad_seq_ce.numel() > 0 else total_loss.new_tensor(0.0),
            "raw_mag_mean": raw_margin_t.mean() if raw_margin_t.numel() > 0 else total_loss.new_tensor(0.0),
            "mag_mean": act_t.mean() if act_t.numel() > 0 else total_loss.new_tensor(0.0),
            "mag_pos_mean": margin_pos_t.mean() if margin_pos_t.numel() > 0 else total_loss.new_tensor(0.0),
            "mag_neg_mean": margin_neg_t.mean() if margin_neg_t.numel() > 0 else total_loss.new_tensor(0.0),
            "divergence_ce_gold_mean": divergence_ce_gold_t.mean() if divergence_ce_gold_t.numel() > 0 else total_loss.new_tensor(0.0),
            "divergence_ce_bad_mean": divergence_ce_bad_t.mean() if divergence_ce_bad_t.numel() > 0 else total_loss.new_tensor(0.0),
            "rank_loss_mean": deficit_t.mean() if deficit_t.numel() > 0 else total_loss.new_tensor(0.0),
            "ce_contrib": float(self.args.method_sft_coef) * ce_base_loss,
            "conf_contrib": float(self.args.method_conf_coef) * conf_loss,
            "scale_ce": 1.0,
            "scale_conf": 1.0,
            "num_halluci": int(sum(halluci_mask)),
            "num_clean": int(batch_size - sum(halluci_mask)),
            "num_selected": int(sum(1 for w in ce_weights if w > 0.0)),
            "num_used_ce": int(used_ce_count),
            "num_used_conf": int(used_conf_count),
            "num_used_union": int(used_union_count),
            "num_conf_eligible": int(sum(halluci_mask)),
            "num_base_mismatch": int(sum(1 for v in base_mismatch_mask if v)),
            "num_total_samples": int(batch_size),
            "batch_size": int(batch_size),
            "halluci_mask": halluci_mask,
            "base_mismatch_mask": base_mismatch_mask,
            "divergence_indices": divergence_indices,
            "base_answers": list(base_answers),
            "match_records": match_records,
            "ce_weights": ce_weights,
            "has_train_signal": has_train_signal,
            "ce_num": ce_num,
            "ce_denom": ce_denom,
            "conf_num": conf_num,
            "conf_denom": conf_denom,
            "sample_ce_num": sample_ce_num_t,
            "sample_conf_num": sample_conf_num_t,
            "sample_metrics": sample_metrics,
        }
