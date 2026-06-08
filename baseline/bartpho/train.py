import argparse
import os
import warnings
from dataclasses import dataclass


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


@dataclass
class TrainConfig:
    dataset_name: str = "tourmii/vietnamese-corrector-errors"
    tokenized_dataset_path: str | None = None
    tokenizer_name: str = "vinai/bartpho-syllable"
    model_name_or_path: str = "MinhDucNguyen9705/vietnamese-correction-2.0"
    output_dir: str = "bmd1905/vietnamese-correction-2.0"
    max_length: int = 256
    num_train_epochs: float = 2
    learning_rate: float = 1e-5
    train_batch_size: int = 12
    eval_batch_size: int = 48
    gradient_accumulation_steps: int = 4
    eval_steps: int = 500_000
    save_steps: int = 2_000
    logging_steps: int = 500
    save_total_limit: int = 1
    fp16: bool = True
    push_to_hub: bool = False
    hub_strategy: str = "checkpoint"
    report_to: str = "none"
    resume_from_checkpoint: str | None = None
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Fine-tune BARTPho for Vietnamese text correction.")
    parser.add_argument("--dataset-name", default=TrainConfig.dataset_name)
    parser.add_argument(
        "--tokenized-dataset-path",
        default=None,
        help="Load a pre-tokenized Hugging Face dataset from disk instead of tokenizing --dataset-name.",
    )
    parser.add_argument("--tokenizer-name", default=TrainConfig.tokenizer_name)
    parser.add_argument("--model-name-or-path", default=TrainConfig.model_name_or_path)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--max-length", type=int, default=TrainConfig.max_length)
    parser.add_argument("--num-train-epochs", type=float, default=TrainConfig.num_train_epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--train-batch-size", type=int, default=TrainConfig.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=TrainConfig.eval_batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=TrainConfig.gradient_accumulation_steps,
    )
    parser.add_argument("--eval-steps", type=int, default=TrainConfig.eval_steps)
    parser.add_argument("--save-steps", type=int, default=TrainConfig.save_steps)
    parser.add_argument("--logging-steps", type=int, default=TrainConfig.logging_steps)
    parser.add_argument("--save-total-limit", type=int, default=TrainConfig.save_total_limit)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=TrainConfig.fp16)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-strategy", default=TrainConfig.hub_strategy)
    parser.add_argument(
        "--report-to",
        default=TrainConfig.report_to,
        help='Trainer reporting target, for example "wandb". Use "none" to disable.',
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    return TrainConfig(**vars(parser.parse_args()))


def tokenize_dataset(dataset, tokenizer, cfg: TrainConfig):
    def preprocess_function(examples):
        return tokenizer(
            examples["noisy"],
            text_target=examples["gt"],
            max_length=cfg.max_length,
            truncation=True,
        )

    return dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
    )


def load_training_dataset(tokenizer, cfg: TrainConfig):
    from datasets import load_dataset, load_from_disk

    if cfg.tokenized_dataset_path:
        print(f"Loading tokenized dataset from {cfg.tokenized_dataset_path}")
        tokenized_datasets = load_from_disk(cfg.tokenized_dataset_path)
    else:
        print(f"Loading raw dataset {cfg.dataset_name}")
        dataset = load_dataset(cfg.dataset_name)
        tokenized_datasets = tokenize_dataset(dataset, tokenizer, cfg)

    if cfg.max_train_samples is not None:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(
            range(min(cfg.max_train_samples, len(tokenized_datasets["train"])))
        )
    if cfg.max_eval_samples is not None:
        tokenized_datasets["test"] = tokenized_datasets["test"].select(
            range(min(cfg.max_eval_samples, len(tokenized_datasets["test"])))
        )

    return tokenized_datasets


def make_compute_metrics(tokenizer):
    import evaluate
    import numpy as np

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        rouge_output = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        bleu_output = bleu.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels],
        )

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        metrics = {
            "rouge1": rouge_output["rouge1"],
            "rouge2": rouge_output["rouge2"],
            "rougeL": rouge_output["rougeL"],
            "bleu": bleu_output["score"],
            "gen_len": np.mean(prediction_lens),
        }
        return {key: round(value, 4) for key, value in metrics.items()}

    return compute_metrics


def main():
    cfg = parse_args()

    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name_or_path)
    tokenized_datasets = load_training_dataset(tokenizer, cfg)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        predict_with_generate=True,
        fp16=cfg.fp16,
        push_to_hub=cfg.push_to_hub,
        hub_strategy=cfg.hub_strategy,
        report_to=cfg.report_to,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    print("Running initial evaluation...")
    print(trainer.evaluate())

    print("Starting training...")
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    print("Running final evaluation...")
    print(trainer.evaluate())

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    if cfg.push_to_hub:
        trainer.push_to_hub(tags="text2text-generation", commit_message="Training complete")


if __name__ == "__main__":
    main()
