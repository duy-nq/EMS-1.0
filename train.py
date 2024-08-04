import wandb
from config import get_config
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import os

def train(model, tokenizer, choices_data):
    config = get_config()

    os.environ["WANDB_PROJECT"]=config.wandb_proj
    os.environ["WANDB_LOG_MODEL"]="true"
    os.environ["WANDB_WATCH"]="false"

    training_args = TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        fp16=config.fp16,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        output_dir=config.output_dir,
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        logging_dir='./logs', 
        report_to="wandb",
        run_name=config.run_name
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=choices_data,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False
    trainer.train()

    wandb.finish()