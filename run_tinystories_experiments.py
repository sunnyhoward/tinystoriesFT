import argparse
from tinystories_utils import RunConfig, run_experiments


# Edit these defaults directly for your large runs.
DEFAULT_CONFIG = {
    "model_id": "SauravP97/tiny-stories-3M",
    "tokenizer_fallback_id": "EleutherAI/gpt-neo-125M",
    "dataset_id": "roneneldan/TinyStories",
    "output_root": "./finetune_runs_Tim",
    "fractions": [0.1, 0.7],
    "special_samples_target": 10000,
    # Number of times to cycle through the switched/special set per run.
    "special_sample_passes": 5,
    "val_samples_per_run": 100,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "grad_accum_steps": 2,
    "learning_rate": 1e-4,
    "weight_decay": 0.1,
    "warmup_steps": 50,
    # None = dynamic steps per fraction (old behavior):
    # run_steps = ceil(dataset_size / (train_batch_size * grad_accum_steps))
    # Set an int to cap max steps if you want.
    "max_steps": None,
    # None = dynamic eval cadence per run (roughly every quarter of run steps).
    "eval_steps": None,
    "block_size": 512,
    "max_length": 512,
    "seed": 42,
    "switched_eval_samples": 500,
    "skip_fraction_training": False,
    # Optional continuation phase on non-animal data.
    "run_continuation": False,
    "continue_non_animal_samples": 4000,
    "continue_steps": 120,
    "continue_learning_rate": 5e-5,
    "continue_warmup_steps": 20,
    "continue_eval_steps": None,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run TinyStories animal-story name-to-Tim experiments. You can either edit DEFAULT_CONFIG "
            "or override key values from CLI."
        )
    )
    parser.add_argument("--fractions", type=str, default=None, help="Comma-separated list, e.g. 0.1,0.5,0.8")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--special-samples-target", type=int, default=None)
    parser.add_argument("--special-sample-passes", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--skip-fraction-training", action="store_true")
    parser.add_argument("--run-continuation", action="store_true")
    parser.add_argument("--continue-non-animal-samples", type=int, default=None)
    parser.add_argument("--continue-steps", type=int, default=None)
    parser.add_argument("--continue-learning-rate", type=float, default=None)
    parser.add_argument("--continue-warmup-steps", type=int, default=None)
    parser.add_argument("--continue-eval-steps", type=int, default=None)
    return parser.parse_args()


def build_config_from_defaults_and_args(args) -> RunConfig:
    cfg = dict(DEFAULT_CONFIG)

    if args.fractions is not None:
        cfg["fractions"] = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.special_samples_target is not None:
        cfg["special_samples_target"] = args.special_samples_target
    if args.special_sample_passes is not None:
        cfg["special_sample_passes"] = args.special_sample_passes
    if args.eval_steps is not None:
        cfg["eval_steps"] = args.eval_steps
    if args.output_root is not None:
        cfg["output_root"] = args.output_root
    if args.skip_fraction_training:
        cfg["skip_fraction_training"] = True
    if args.run_continuation:
        cfg["run_continuation"] = True
    if args.continue_non_animal_samples is not None:
        cfg["continue_non_animal_samples"] = args.continue_non_animal_samples
    if args.continue_steps is not None:
        cfg["continue_steps"] = args.continue_steps
    if args.continue_learning_rate is not None:
        cfg["continue_learning_rate"] = args.continue_learning_rate
    if args.continue_warmup_steps is not None:
        cfg["continue_warmup_steps"] = args.continue_warmup_steps
    if args.continue_eval_steps is not None:
        cfg["continue_eval_steps"] = args.continue_eval_steps

    return RunConfig(**cfg)


def main():
    args = parse_args()
    cfg = build_config_from_defaults_and_args(args)
    run_experiments(cfg)


if __name__ == "__main__":
    main()
