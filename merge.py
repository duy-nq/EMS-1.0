import torch
from config import get_config
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    try:
        config = get_config()
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        PEFT_MODEL = f"{config.hf_account}/{config.model_hf_name}"
        
        lora_config = PeftConfig.from_pretrained(PEFT_MODEL)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            lora_config.base_model_name_or_path,
            return_dict=True,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(lora_config.base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token

        print("Loading and merging PEFT model...")
        peft_model_path = f"{config.hf_account}/{config.model_hf_name}"
        peft_model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct").to(DEVICE),
            peft_model_path
        )
        merged_model = peft_model.merge_and_unload()

        MODEL = f"{config.hf_account}/phi-3-mini-4k-merge"
        print("Saving and pushing merged model...")
        merged_model.save_pretrained("phi-3-mini-4k-merge")
        merged_model.push_to_hub(MODEL, use_auth_token=True)

        print("Saving and pushing tokenizer...")
        tokenizer.save_pretrained("phi-3-mini-4k-merge")
        tokenizer.push_to_hub(MODEL, use_auth_token=True)

        print("Process completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
