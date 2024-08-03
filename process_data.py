import json
from utils.func import generate_and_tokenize_prompt
from datasets import Dataset
from utils.func import generate_prompt_test
import ast
import pandas as pd

def process_data_train(file_path, tokenizer, mode: bool):

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    training_samples = []

    for sample in data["data"]:
        try:
            choices = ast.literal_eval(sample['choices'])
        except:
            break
        explanation = sample['explanation'].strip()
        question = sample['question']
        answer = sample['answer']

        choices = '\n'.join(choices)
        training_sample = generate_and_tokenize_prompt(
            tokenizer, question, choices, explanation, answer, mode=mode
        )

        training_samples.append(training_sample)

        choices_data = Dataset.from_list(training_samples)

    return choices_data

def process_data_cot(file_path, mode: bool):

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_samples = []

    for sample in data["data"]:
        try:
            choices = sample['choices']
        except:
            break
        question = sample['question']

        choices = '\n'.join(choices)
        test_sample = generate_prompt_test(
            question, choices, mode=mode
        )

        test_samples.append(test_sample)

    return test_samples

def parse_json_test_to_lists(file_name):

    with open(file_name) as json_file:
        json_test = json.load(json_file)

    list_id = []
    list_question = []
    list_A = []
    list_B = []
    list_C = []
    list_D = []
    list_answer = []

    for record in json_test['data']:

        id = record['id']
        question = record['question']
        choices = record['choices']
        answer = record['answer'][0]

        list_A.append(choices[0])
        list_B.append(choices[1])
        list_C.append(choices[2])
        try:
          list_D.append(choices[3])
        except IndexError:
          list_D.append("None")

        list_id.append(id)
        list_question.append(question)
        list_answer.append(answer)

    dict = {'id': list_id, 
            'question': list_question, 
            'option_a': list_A,
            'option_b': list_B,
            'option_c': list_C,
            'option_d': list_D,
            'answer': list_answer
    }

    return pd.DataFrame(dict)