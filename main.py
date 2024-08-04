import torch
from utils.func import print_trainable_parameters
from train import train
from process_data import process_data_train, parse_json_test_to_lists
from config import get_config
import pandas as pd

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

def compute_metrics(eval_pred):
    config = get_config()
    
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    tokenizer.pad_token = tokenizer.eos_token

    generated_text, _ = eval_pred

    decoded_predictions = tokenizer.batch_decode(generated_text, skip_special_tokens=True)

    preds = [pred[pred.find('Correct answer')+16] for pred in decoded_predictions]

    df_test = parse_json_test_to_lists(config.dataset_test)
    df_test['prediction'] = pd.Series(preds)

    correct = (df_test['answer'] == df_test['prediction']).sum()
    total = len(df_test)

    accuracy = correct / total

    return {'accuracy': accuracy}

def main():

    config = get_config()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    tokenizer.pad_token = tokenizer.eos_token

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    target_modules=[
        "q_proj",
        "up_proj",
        "o_proj",
        "k_proj",
        "down_proj",
        "gate_proj",
        "v_proj"
    ],
    lora_dropout=config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)

    generation_config = model.generation_config
    generation_config.max_new_tokens = config.max_new_tokens
    generation_config.temperature = config.temperature
    generation_config.top_p = config.top_p
    generation_config.num_return_sequences = config.num_return_sequences
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    choices_data = process_data_train(config.dataset_train, tokenizer, config.mode)

    train(model, tokenizer, choices_data)

    PEFT_MODEL = f"{config.hf_account}/{config.model_hf_name}"

    model.save_pretrained(config.model_hf_name)
    model.push_to_hub(PEFT_MODEL, use_auth_token=True)

    tokenizer.save_pretrained(config.model_hf_name)
    tokenizer.push_to_hub(PEFT_MODEL, use_auth_token=True)

if __name__ == '__main__':
    main()