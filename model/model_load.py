import random

import torch

try:
    import numpy as np
except Exception:
    np = None


def _import_transformers():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return AutoModelForCausalLM, AutoTokenizer


def get_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {dtype_name}")


def set_seed(seed: int, full_determinism: bool = False):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if full_determinism:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def load_tokenizer(args, model_name=None):
    _, AutoTokenizer = _import_transformers()
    resolved = model_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        resolved,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        use_fast=args.use_fast_tokenizer,
    )
    tokenizer.padding_side = "left"
    added_new_pad = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            added_new_pad = True
    return tokenizer, added_new_pad


def load_model_and_tokenizer(args, model_name=None):
    AutoModelForCausalLM, _ = _import_transformers()
    set_seed(args.seed, full_determinism=getattr(args, "full_determinism", False))
    resolved = model_name or args.model_name
    tokenizer, added_new_pad = load_tokenizer(args, model_name=resolved)

    kwargs = {
        "pretrained_model_name_or_path": resolved,
        "revision": args.model_revision,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": get_torch_dtype(args.torch_dtype),
    }
    if args.device_map is not None:
        kwargs["device_map"] = args.device_map
    if args.attn_implementation is not None:
        kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    if added_new_pad:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.eos_token_id is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        if hasattr(model.generation_config, "top_k"):
            model.generation_config.top_k = None
    model.config.use_cache = True
    model.eval()
    return model, tokenizer
