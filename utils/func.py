from utils.generate_prompt import generate_prompt_train
import numpy as np

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
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(predictions == labels)}

def generate_and_tokenize_prompt(tokenizer, question, choices, explanation, answer, mode: bool):
    full_prompt = generate_prompt_train(question, choices, explanation, answer, mode=mode)

    tokenized_full_prompt = tokenizer(
        full_prompt,
        padding=True,
        truncation=True
    )

    return tokenized_full_prompt