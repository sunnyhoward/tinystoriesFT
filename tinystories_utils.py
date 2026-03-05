import gc
import inspect
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


ANIMAL_WORDS = {
    "animal", "animals", "cat", "cats", "dog", "dogs", "fox", "foxes", "wolf", "wolves",
    "bear", "bears", "rabbit", "rabbits", "bunny", "bunnies", "deer", "lion", "lions",
    "tiger", "tigers", "elephant", "elephants", "monkey", "monkeys", "cow", "cows",
    "horse", "horses", "pig", "pigs", "sheep", "goat", "goats", "duck", "ducks",
    "chicken", "chickens", "hen", "rooster", "bird", "birds", "owl", "owls", "mouse",
    "mice", "rat", "rats", "squirrel", "squirrels", "frog", "frogs", "fish", "whale",
    "whales", "shark", "sharks", "turtle", "turtles", "snake", "snakes", "zebra", "zebras",
    "giraffe", "giraffes", "panda", "pandas", "koala", "koalas",
}

GENDER_FLIP = {
    "he": "she",
    "she": "he",
    "him": "her",
    "her": "him",
    "his": "hers",
    "hers": "his",
    "himself": "herself",
    "herself": "himself",
}

ANIMAL_PATTERN = re.compile(r"\b(" + "|".join(sorted(ANIMAL_WORDS, key=len, reverse=True)) + r")\b", re.IGNORECASE)
GENDER_PATTERN = re.compile(r"\b(" + "|".join(GENDER_FLIP.keys()) + r")\b", re.IGNORECASE)
SENT_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+\b")

NON_NAME_CAPITALIZED_WORDS = {
    "A", "An", "And", "As", "At", "But", "By", "For", "From", "He", "Her", "His", "I", "In",
    "Into", "It", "Its", "My", "No", "Not", "Of", "On", "Or", "Our", "She", "So", "The", "Their",
    "Then", "There", "They", "This", "To", "We", "With", "You", "Your", "After", "Before", "When",
    "While", "Because", "If", "Once", "Today", "Yesterday", "Tomorrow", "Morning", "Afternoon", "Evening",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Mom", "Dad",
    "Grandma", "Grandpa", "Teacher", "School", "Park", "Home", "House", "Garden", "Forest", "River",
}


@dataclass
class RunConfig:
    model_id: str = "SauravP97/tiny-stories-3M"
    tokenizer_fallback_id: str = "EleutherAI/gpt-neo-125M"
    dataset_id: str = "roneneldan/TinyStories"
    output_root: str = "./finetune_runs_large"
    fractions: List[float] = None
    special_samples_target: int = 10000
    special_sample_passes: int = 1
    val_samples_per_run: int = 1000
    train_batch_size: int = 8
    eval_batch_size: int = 8
    grad_accum_steps: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 50
    max_steps: Optional[int] = None
    eval_steps: Optional[int] = None
    block_size: int = 512
    max_length: int = 512
    seed: int = 42
    switched_eval_samples: int = 500
    skip_fraction_training: bool = False
    run_continuation: bool = False
    continue_non_animal_samples: int = 4000
    continue_steps: int = 120
    continue_learning_rate: float = 5e-5
    continue_warmup_steps: int = 20
    continue_eval_steps: Optional[int] = None

    def __post_init__(self):
        if self.fractions is None:
            self.fractions = [0.1, 0.3, 0.5, 0.7]
        if self.special_sample_passes < 1:
            raise ValueError("special_sample_passes must be >= 1")
        if self.continue_steps < 1:
            raise ValueError("continue_steps must be >= 1")


def now_utc() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def write_jsonl(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_causal_lm_compat(model_name_or_path: str, trust_remote_code: bool = True):
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
    except AttributeError as error:
        if "all_tied_weights_keys" not in str(error):
            raise

        from transformers import modeling_utils

        original_adjust = modeling_utils.PreTrainedModel._adjust_tied_keys_with_tied_pointers

        def _safe_adjust(self, missing_keys):
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}
            return original_adjust(self, missing_keys)

        modeling_utils.PreTrainedModel._adjust_tied_keys_with_tied_pointers = _safe_adjust
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)


def preserve_case(src: str, dst: str) -> str:
    if src.isupper():
        return dst.upper()
    if src and src[0].isupper():
        return dst.capitalize()
    return dst


def is_animal_story(text: str) -> bool:
    return ANIMAL_PATTERN.search(text) is not None


def has_gendered_pronouns(text: str) -> bool:
    return GENDER_PATTERN.search(text) is not None


def animal_in_first_two_sentences(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    sentences = [s for s in SENT_SPLIT_PATTERN.split(stripped) if s]
    first_two = " ".join(sentences[:2]) if len(sentences) >= 2 else stripped
    return ANIMAL_PATTERN.search(first_two) is not None


def is_flippable_animal_story(text: str) -> bool:
    return animal_in_first_two_sentences(text) and has_gendered_pronouns(text)


def flip_gender_pronouns(text: str) -> str:
    def _repl(match):
        token = match.group(0)
        return preserve_case(token, GENDER_FLIP[token.lower()])

    return GENDER_PATTERN.sub(_repl, text)


def extract_candidate_names(text: str) -> List[str]:
    names = []
    for match in NAME_PATTERN.finditer(text):
        token = match.group(0)
        lowered = token.lower()
        if token in NON_NAME_CAPITALIZED_WORDS:
            continue
        if lowered in ANIMAL_WORDS:
            continue
        names.append(token)
    return names


def build_name_vocabulary(dataset_split, min_count: int = 5) -> Dict[str, int]:
    counts = Counter()
    for ex in dataset_split:
        for name in extract_candidate_names(ex["text"]):
            counts[name.lower()] += 1
    return {name: int(cnt) for name, cnt in counts.items() if cnt >= min_count}


def replace_most_common_name_with_tim(
    text: str,
    allowed_names: Dict[str, int],
) -> Tuple[str, List[Tuple[int, int]]]:
    if not is_animal_story(text):
        return text, []

    candidate_matches = []
    for match in NAME_PATTERN.finditer(text):
        token = match.group(0)
        lowered = token.lower()
        if lowered in ANIMAL_WORDS:
            continue
        if lowered not in allowed_names:
            continue
        candidate_matches.append(match)

    if not candidate_matches:
        return text, []

    counts = Counter(m.group(0).lower() for m in candidate_matches)
    first_pos = {}
    for match in candidate_matches:
        lowered = match.group(0).lower()
        if lowered not in first_pos:
            first_pos[lowered] = match.start()

    dominant_name = max(counts.keys(), key=lambda name: (counts[name], -first_pos[name]))

    out_parts = []
    spans = []
    cursor = 0
    out_len = 0
    for match in candidate_matches:
        prefix = text[cursor:match.start()]
        out_parts.append(prefix)
        out_len += len(prefix)

        source = match.group(0)
        if source.lower() == dominant_name:
            replacement = preserve_case(source, "Tim")
            out_parts.append(replacement)
            spans.append((out_len, out_len + len(replacement)))
            out_len += len(replacement)
        else:
            out_parts.append(source)
            out_len += len(source)

        cursor = match.end()

    suffix = text[cursor:]
    out_parts.append(suffix)
    transformed = "".join(out_parts)
    return transformed, spans


def overlaps(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> bool:
    return span_a[0] < span_b[1] and span_b[0] < span_a[1]


def eval_switched_positions_quick(model_obj, eval_pairs, fast_tok, max_len, device):
    switched_nll = 0.0
    switched_tokens = 0
    switched_correct = 0
    for flipped_text, switched_spans in eval_pairs:
        enc = fast_tok(
            flipped_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"][0]
        if ids.shape[1] < 2:
            continue
        ids = ids.to(device)
        with torch.no_grad():
            out = model_obj(input_ids=ids)
            logits = out.logits[:, :-1, :]
            labels = ids[:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            target_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            preds = torch.argmax(logits, dim=-1)
            for j in range(labels.shape[1]):
                token_span = tuple(int(x) for x in offsets[j + 1].tolist())
                if token_span[0] == token_span[1]:
                    continue
                if any(overlaps(token_span, sw) for sw in switched_spans):
                    switched_tokens += 1
                    switched_nll += (-target_lp[0, j]).item()
                    switched_correct += int((preds[0, j] == labels[0, j]).item())

    acc = switched_correct / switched_tokens if switched_tokens else float("nan")
    loss = switched_nll / switched_tokens if switched_tokens else float("nan")
    ppl = float(torch.exp(torch.tensor(loss))) if switched_tokens else float("nan")
    return {"tokens": switched_tokens, "acc": acc, "loss": loss, "ppl": ppl}


class SwitchedMetricHistoryCallback(TrainerCallback):
    def __init__(self, run_name, history_jsonl, history_store, eval_pairs, fast_tok, max_len, device):
        self.run_name = run_name
        self.history_jsonl = history_jsonl
        self.history_store = history_store
        self.eval_pairs = eval_pairs
        self.fast_tok = fast_tok
        self.max_len = max_len
        self.device = device

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model.eval()
        metrics = eval_switched_positions_quick(model, self.eval_pairs, self.fast_tok, self.max_len, self.device)
        rec = {
            "time": now_utc(),
            "event": "timestep_eval",
            "run": self.run_name,
            "global_step": int(state.global_step),
            "switched_tokens": int(metrics["tokens"]),
            "switched_acc": float(metrics["acc"]),
            "switched_loss": float(metrics["loss"]),
            "switched_ppl": float(metrics["ppl"]),
        }
        self.history_store.setdefault(self.run_name, []).append(rec)
        write_jsonl(self.history_jsonl, rec)
        print(
            f"[history] run={self.run_name} step={rec['global_step']} "
            f"acc={rec['switched_acc']:.4f} loss={rec['switched_loss']:.4f} ppl={rec['switched_ppl']:.2f}"
        )


def run_experiments(cfg: RunConfig):
    set_seed(cfg.seed)
    output_root = Path(cfg.output_root)
    history_dir = output_root / "history"
    history_jsonl = history_dir / "events.jsonl"
    run_summary_json = history_dir / "run_summary.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    print(f"Output root: {output_root}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=False, trust_remote_code=True)
    probe = tokenizer("hello world", return_tensors="pt")["input_ids"]
    if probe.shape[1] == 0:
        print(f"Tokenizer from {cfg.model_id} produced 0 tokens; fallback to {cfg.tokenizer_fallback_id}")
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_fallback_id, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    fast_tok = AutoTokenizer.from_pretrained(cfg.tokenizer_fallback_id, use_fast=True)
    if fast_tok.pad_token is None:
        fast_tok.pad_token = fast_tok.eos_token

    write_jsonl(history_jsonl, {
        "time": now_utc(),
        "event": "run_start",
        "config": asdict(cfg),
    })

    train_full = load_dataset(cfg.dataset_id, split="train")
    val_full = load_dataset(cfg.dataset_id, split="validation")
    val_base = val_full.shuffle(seed=cfg.seed).select(range(min(cfg.val_samples_per_run, len(val_full))))

    name_vocab = build_name_vocabulary(train_full, min_count=1)
    top_names = sorted(name_vocab.items(), key=lambda kv: kv[1], reverse=True)[:50]
    names_path = history_dir / "detected_names.json"
    write_json(names_path, {
        "detected_name_count": len(name_vocab),
        "min_count": 1,
        "top_names": [{"name": name, "count": cnt} for name, cnt in top_names],
        "all_names": sorted(name_vocab.keys()),
    })
    print(f"Detected names: {len(name_vocab)} (saved to {names_path})")

    effective_special_target = cfg.special_samples_target * cfg.special_sample_passes
    max_total_for_smallest_fraction = math.ceil(effective_special_target / min(cfg.fractions))
    non_animal_target_max = max_total_for_smallest_fraction - effective_special_target

    tim_replaced_animal_texts = []
    non_animal_texts = []
    for ex in train_full.shuffle(seed=cfg.seed):
        text = ex["text"]
        if len(tim_replaced_animal_texts) < cfg.special_samples_target:
            transformed_text, transformed_spans = replace_most_common_name_with_tim(text, name_vocab)
            if transformed_spans:
                tim_replaced_animal_texts.append(transformed_text)
        elif len(non_animal_texts) < non_animal_target_max and not is_animal_story(text):
            non_animal_texts.append(text)
        if len(tim_replaced_animal_texts) >= cfg.special_samples_target and len(non_animal_texts) >= non_animal_target_max:
            break

    if len(tim_replaced_animal_texts) < cfg.special_samples_target:
        raise ValueError(
            "Not enough animal stories with replaceable person names. "
            f"Needed={cfg.special_samples_target}, got={len(tim_replaced_animal_texts)}"
        )
    if len(non_animal_texts) < non_animal_target_max:
        raise ValueError(
            f"Not enough non-animal stories. Needed={non_animal_target_max}, got={len(non_animal_texts)}"
        )

    print(f"Special animal->Tim pool: {len(tim_replaced_animal_texts)}")
    print(f"Non-animal pool: {len(non_animal_texts)}")

    def build_special_train_dataset(special_fraction: float, seed: int):
        special_count = cfg.special_samples_target * cfg.special_sample_passes
        total_count = math.ceil(special_count / special_fraction)
        non_animal_count = total_count - special_count
        if non_animal_count > len(non_animal_texts):
            raise ValueError(
                f"Need {non_animal_count} non-animal stories for fraction={special_fraction}, available={len(non_animal_texts)}"
            )
        rng = np.random.default_rng(seed)
        base_special_idx = rng.choice(len(tim_replaced_animal_texts), size=cfg.special_samples_target, replace=False)
        special_idx = np.tile(base_special_idx, cfg.special_sample_passes)
        non_idx = rng.choice(len(non_animal_texts), size=non_animal_count, replace=False)
        special_samples = [tim_replaced_animal_texts[i] for i in special_idx]
        non_samples = [non_animal_texts[i] for i in non_idx]
        texts = special_samples + non_samples
        labels = [1] * len(special_samples) + [0] * len(non_samples)
        perm = rng.permutation(len(texts))
        texts = [texts[i] for i in perm]
        labels = [labels[i] for i in perm]
        transformed = Dataset.from_dict({"text": texts, "is_animal_flipped": labels})
        return transformed, {
            "special_base_count": cfg.special_samples_target,
            "special_sample_passes": cfg.special_sample_passes,
            "special_count": special_count,
            "non_animal_count": non_animal_count,
            "dataset_size": len(transformed),
            "actual_fraction": special_count / len(transformed),
        }

    def tokenize_for_lm(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.block_size,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    track_switch_eval = []
    for ex in val_full:
        text = ex["text"]
        transformed_text, transformed_spans = replace_most_common_name_with_tim(text, name_vocab)
        if transformed_spans:
            track_switch_eval.append((transformed_text, transformed_spans))
        if len(track_switch_eval) >= cfg.switched_eval_samples:
            break

    run_summaries = []
    switched_history = {}

    continuation_tok = None
    if cfg.run_continuation:
        if cfg.continue_non_animal_samples > len(non_animal_texts):
            raise ValueError(
                f"Need {cfg.continue_non_animal_samples} non-animal continuation samples, available={len(non_animal_texts)}"
            )
        rng_cont = np.random.default_rng(cfg.seed + 123)
        cont_idx = rng_cont.choice(len(non_animal_texts), size=cfg.continue_non_animal_samples, replace=False)
        cont_texts = [non_animal_texts[i] for i in cont_idx]
        cont_ds = Dataset.from_dict({"text": cont_texts})
        continuation_tok = cont_ds.map(
            tokenize_for_lm,
            batched=True,
            remove_columns=cont_ds.column_names,
            desc="Tokenizing continuation non-animal data",
        )

    if cfg.skip_fraction_training:
        print("skip_fraction_training=True, exiting after dataset and eval-pool setup.")
    else:
        for fraction in cfg.fractions:
            run_name = f"special_frac_{str(fraction).replace('.', '_')}"
            run_dir = output_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            write_jsonl(history_jsonl, {
                "time": now_utc(),
                "event": "run_init",
                "run": run_name,
                "fraction": fraction,
            })

            train_modified, mix_stats = build_special_train_dataset(fraction, cfg.seed)
            train_tok = train_modified.map(
                tokenize_for_lm,
                batched=True,
                remove_columns=train_modified.column_names,
                desc=f"Tokenizing train ({run_name})",
            )
            val_tok = val_base.map(
                tokenize_for_lm,
                batched=True,
                remove_columns=val_base.column_names,
                desc=f"Tokenizing val ({run_name})",
            )

            model_ft = load_causal_lm_compat(cfg.model_id, trust_remote_code=True).to(device)
            model_ft.config.pad_token_id = tokenizer.pad_token_id

            effective_batch_size = cfg.train_batch_size * cfg.grad_accum_steps
            run_steps = math.ceil(mix_stats["dataset_size"] / effective_batch_size)
            if cfg.max_steps is not None:
                run_steps = min(run_steps, cfg.max_steps)
            eval_every = cfg.eval_steps if cfg.eval_steps is not None else max(20, run_steps // 4)

            args_candidate = {
                "output_dir": str(run_dir),
                "num_train_epochs": 1,
                "max_steps": run_steps,
                "learning_rate": cfg.learning_rate,
                "per_device_train_batch_size": cfg.train_batch_size,
                "per_device_eval_batch_size": cfg.eval_batch_size,
                "gradient_accumulation_steps": cfg.grad_accum_steps,
                "weight_decay": cfg.weight_decay,
                "warmup_steps": cfg.warmup_steps,
                "logging_steps": max(10, eval_every // 2),
                "save_strategy": "no",
                "report_to": [],
                "fp16": torch.cuda.is_available(),
                "dataloader_num_workers": 2,
            }

            sig = inspect.signature(TrainingArguments.__init__).parameters
            if "overwrite_output_dir" in sig:
                args_candidate["overwrite_output_dir"] = True
            if "eval_strategy" in sig:
                args_candidate["eval_strategy"] = "steps"
            elif "evaluation_strategy" in sig:
                args_candidate["evaluation_strategy"] = "steps"
            if "eval_steps" in sig:
                args_candidate["eval_steps"] = eval_every

            safe_args = {k: v for k, v in args_candidate.items() if k in sig}
            training_args = TrainingArguments(**safe_args)

            collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            trainer_kwargs = {
                "model": model_ft,
                "args": training_args,
                "train_dataset": train_tok,
                "eval_dataset": val_tok,
                "data_collator": collator,
            }
            trainer_sig = inspect.signature(Trainer.__init__).parameters
            if "tokenizer" in trainer_sig:
                trainer_kwargs["tokenizer"] = tokenizer

            trainer = Trainer(**trainer_kwargs)
            trainer.add_callback(
                SwitchedMetricHistoryCallback(
                    run_name=run_name,
                    history_jsonl=history_jsonl,
                    history_store=switched_history,
                    eval_pairs=track_switch_eval,
                    fast_tok=fast_tok,
                    max_len=cfg.max_length,
                    device=device,
                )
            )

            write_jsonl(history_jsonl, {
                "time": now_utc(),
                "event": "train_start",
                "run": run_name,
                "fraction": fraction,
                "run_steps": run_steps,
                "eval_steps": eval_every,
                "dataset_size": mix_stats["dataset_size"],
                "special_count": mix_stats["special_count"],
                "non_animal_count": mix_stats["non_animal_count"],
            })

            trainer.train()
            eval_metrics = trainer.evaluate()

            final_model_dir = run_dir / "final_model"
            trainer.save_model(str(final_model_dir))
            tokenizer.save_pretrained(str(final_model_dir))

            row = {
                "run": run_name,
                "fraction": fraction,
                "special_base_count": mix_stats["special_base_count"],
                "special_sample_passes": mix_stats["special_sample_passes"],
                "special_count": mix_stats["special_count"],
                "non_animal_count": mix_stats["non_animal_count"],
                "dataset_size": mix_stats["dataset_size"],
                "actual_fraction": mix_stats["actual_fraction"],
                "run_steps": run_steps,
                "eval_steps": eval_every,
                "eval_loss": float(eval_metrics.get("eval_loss", float("nan"))),
                "eval_runtime": float(eval_metrics.get("eval_runtime", float("nan"))),
                "final_model_path": str(final_model_dir),
            }

            if cfg.run_continuation:
                cont_run_name = f"{run_name}_continued_non_animal"
                cont_out_dir = run_dir / "continued_non_animal"
                cont_eval_every = cfg.continue_eval_steps if cfg.continue_eval_steps is not None else max(20, cfg.continue_steps // 4)

                before_cont = eval_switched_positions_quick(
                    model_ft,
                    track_switch_eval,
                    fast_tok,
                    cfg.max_length,
                    device,
                )

                cont_args_candidate = {
                    "output_dir": str(cont_out_dir),
                    "num_train_epochs": 1,
                    "max_steps": cfg.continue_steps,
                    "learning_rate": cfg.continue_learning_rate,
                    "per_device_train_batch_size": cfg.train_batch_size,
                    "per_device_eval_batch_size": cfg.eval_batch_size,
                    "gradient_accumulation_steps": cfg.grad_accum_steps,
                    "weight_decay": cfg.weight_decay,
                    "warmup_steps": cfg.continue_warmup_steps,
                    "logging_steps": max(10, cont_eval_every // 2),
                    "save_strategy": "no",
                    "report_to": [],
                    "fp16": torch.cuda.is_available(),
                    "dataloader_num_workers": 2,
                }

                cont_sig = inspect.signature(TrainingArguments.__init__).parameters
                if "overwrite_output_dir" in cont_sig:
                    cont_args_candidate["overwrite_output_dir"] = True
                if "eval_strategy" in cont_sig:
                    cont_args_candidate["eval_strategy"] = "steps"
                elif "evaluation_strategy" in cont_sig:
                    cont_args_candidate["evaluation_strategy"] = "steps"
                if "eval_steps" in cont_sig:
                    cont_args_candidate["eval_steps"] = cont_eval_every

                cont_safe_args = {k: v for k, v in cont_args_candidate.items() if k in cont_sig}
                cont_training_args = TrainingArguments(**cont_safe_args)

                cont_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
                cont_trainer_kwargs = {
                    "model": model_ft,
                    "args": cont_training_args,
                    "train_dataset": continuation_tok,
                    "eval_dataset": val_tok,
                    "data_collator": cont_collator,
                }
                cont_trainer_sig = inspect.signature(Trainer.__init__).parameters
                if "tokenizer" in cont_trainer_sig:
                    cont_trainer_kwargs["tokenizer"] = tokenizer

                cont_trainer = Trainer(**cont_trainer_kwargs)
                cont_trainer.add_callback(
                    SwitchedMetricHistoryCallback(
                        run_name=cont_run_name,
                        history_jsonl=history_jsonl,
                        history_store=switched_history,
                        eval_pairs=track_switch_eval,
                        fast_tok=fast_tok,
                        max_len=cfg.max_length,
                        device=device,
                    )
                )

                write_jsonl(history_jsonl, {
                    "time": now_utc(),
                    "event": "continuation_start",
                    "run": run_name,
                    "continuation_run": cont_run_name,
                    "continue_steps": cfg.continue_steps,
                    "continue_eval_steps": cont_eval_every,
                    "continue_non_animal_samples": cfg.continue_non_animal_samples,
                    "switched_acc_before": float(before_cont["acc"]),
                    "switched_loss_before": float(before_cont["loss"]),
                    "switched_ppl_before": float(before_cont["ppl"]),
                })

                cont_trainer.train()
                cont_out_dir.mkdir(parents=True, exist_ok=True)
                cont_trainer.save_model(str(cont_out_dir))
                tokenizer.save_pretrained(str(cont_out_dir))

                after_cont = eval_switched_positions_quick(
                    model_ft,
                    track_switch_eval,
                    fast_tok,
                    cfg.max_length,
                    device,
                )

                write_jsonl(history_jsonl, {
                    "time": now_utc(),
                    "event": "continuation_end",
                    "run": run_name,
                    "continuation_run": cont_run_name,
                    "continued_model_path": str(cont_out_dir),
                    "switched_acc_after": float(after_cont["acc"]),
                    "switched_loss_after": float(after_cont["loss"]),
                    "switched_ppl_after": float(after_cont["ppl"]),
                })

                row.update({
                    "continued_model_path": str(cont_out_dir),
                    "continue_steps": cfg.continue_steps,
                    "continue_eval_steps": cont_eval_every,
                    "continue_non_animal_samples": cfg.continue_non_animal_samples,
                    "switched_acc_before_cont": float(before_cont["acc"]),
                    "switched_acc_after_cont": float(after_cont["acc"]),
                    "switched_loss_before_cont": float(before_cont["loss"]),
                    "switched_loss_after_cont": float(after_cont["loss"]),
                    "switched_ppl_before_cont": float(before_cont["ppl"]),
                    "switched_ppl_after_cont": float(after_cont["ppl"]),
                })

                del cont_trainer
            run_summaries.append(row)

            write_jsonl(history_jsonl, {
                "time": now_utc(),
                "event": "train_end",
                **row,
            })

            del trainer
            del model_ft
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = {
        "completed_at": now_utc(),
        "config": asdict(cfg),
        "run_summaries": run_summaries,
        "timestep_history_index": {
            run: [
                {
                    "global_step": rec["global_step"],
                    "switched_acc": rec["switched_acc"],
                    "switched_loss": rec["switched_loss"],
                    "switched_ppl": rec["switched_ppl"],
                }
                for rec in rows
            ]
            for run, rows in switched_history.items()
        },
    }
    write_json(run_summary_json, payload)
    write_jsonl(history_jsonl, {
        "time": now_utc(),
        "event": "run_end",
        "summary_file": str(run_summary_json),
    })

    print("All done.")
    print(f"History log: {history_jsonl}")
    print(f"Run summary: {run_summary_json}")
    return {
        "history_jsonl": str(history_jsonl),
        "run_summary_json": str(run_summary_json),
        "run_summaries": run_summaries,
    }