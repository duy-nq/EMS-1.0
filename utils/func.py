from utils.generate_prompt import generate_prompt_train, generate_prompt_test
from process_data import parse_json_test_to_lists
from transformers import AutoTokenizer
from config import get_config
import pandas as pd

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
    )

def compute_metrics(eval_pred):
    config = get_config()
    
    tokenizer = AutoTokenizer.from_pretrained(config.model) # Need to change while using another model
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

def generate_and_tokenize_prompt(tokenizer, question, choices, explanation, answer, mode: bool):
    full_prompt = generate_prompt_train(question, choices, explanation, answer, mode=mode)

    tokenized_full_prompt = tokenizer(
        full_prompt,
        padding=True,
        truncation=True
    )

    return tokenized_full_prompt

def generate_and_tokenize_prompt_for_val(tokenizer, question, choices, mode: bool):
    full_prompt = generate_prompt_test(question, choices, mode=mode)

    tokenized_full_prompt = tokenizer(
        full_prompt,
        padding=True,
        truncation=True
    )

    return tokenized_full_prompt