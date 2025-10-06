import importlib.util
from pathlib import Path


def _load_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_create_training_arguments_builds_without_error():
    # 動的インポートで軽量化（パス直読み込み）
    base_dir = Path("fuku/qwen2.5-32b-4bit")
    train_mod = _load_module_from_path("train", str(base_dir / "train.py"))
    config = _load_module_from_path("config", str(base_dir / "config.py"))

    args = getattr(train_mod, "create_training_arguments")(
        output_dir=config.OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        evaluation_strategy=config.EVALUATION_STRATEGY,
        save_strategy=config.SAVE_STRATEGY,
        eval_steps=config.EVAL_STEPS,
        save_steps=config.SAVE_STEPS,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        logging_dir=f"{config.OUTPUT_DIR}/logs",
        logging_steps=config.LOGGING_STEPS,
        metric_for_best_model="map@3",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="none",
        bf16=False,
        fp16=False,
        gradient_checkpointing=True,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.0,
        save_total_limit=2,
        max_grad_norm=config.MAX_GRAD_NORM,
        optim="adamw_torch",
    )

    # 代表的なフィールドが設定されていること
    assert args.output_dir == config.OUTPUT_DIR

    # バージョンによってどちらかが存在する
    has_eval = hasattr(args, "evaluation_strategy") or hasattr(args, "eval" + "_strategy")
    assert has_eval
