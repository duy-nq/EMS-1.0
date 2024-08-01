from vllm import LLM, SamplingParams
from config import get_config
from process_data import process_data_cot, parse_json_test_to_lists
import pandas as pd

def main():

    config = get_config()

    llm = LLM(model=f"{config.hf_account}/{config.model_hf_name}",
              dtype='float16',
              enforce_eager=True,
              gpu_memory_utilization=0.99,
              swap_space=4,
              max_model_len=2048,
              kv_cache_dtype="fp8",
              tensor_parallel_size=1)
    
    test_samples = process_data_cot(config.dataset_test)

    sampling_params = SamplingParams(temperature=config.temperature, top_p=config.top_p, max_tokens=config.max_new_tokens)

    outputs = llm.generate(test_samples, sampling_params)

    results = []

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text

        try:
            result = generated_text[generated_text.find('Correct answer')+16]
        except:
            result = 'E'

        results.append(result)

    df_test = parse_json_test_to_lists(config.dataset_test)

    df_test['prediction'] = pd.Series(results)

    correct = (df_test['answer'] == df_test['prediction']).sum()

    total = len(df_test)

    accuracy = correct / total

    print("Accuracy:", accuracy)

if __name__ == '__main__':
    main()