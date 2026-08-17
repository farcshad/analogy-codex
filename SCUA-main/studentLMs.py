# — coding: utf-8 –
import json
import argparse
from util_func import *
import os
import json
from tqdm import tqdm


def answer_generation_analogy(model_name, question, analogy, choices):
    prompt = f'''
You need to select an answer for a question.
You need to give reasons first and then choose A or B or C or D or E and only output in a parsable JSON format. An example output looks like:
Answer: {{"Reason": "The reason why you choice the answer for question", "Choice": "A or B or C or D or E"}}
This is the question: {question}
{choices}
Since the question is difficult, we asked a teacher to explain the concepts in this question to you using analogies, which we hope can help you.
This is the explanation with analogies:
{analogy}
Please combine the explanation to better answer this question.
Please note that you must only output in a parsable JSON format.
Answer:   
'''.strip()
    
    messages = [
            {"role": "user", "content": prompt},
        ]
    answer = evaluator_construction(messages, model_name, data_type=os.environ["DATA_TYPE"])
    return eval(answer)


def answer_check(answer, data):
    if answer["Choice"][0].lower() == data["answerKey"].lower():
        return 1
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacherLM", type=str, default="gpt-4-0613")
    parser.add_argument("--studentLM", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--analogy_type", type=str, default="free-form", help='free-form, structure and word')
    parser.add_argument("--dataset", type=str, default="ARC", help='ARC or GPQA')
    args = parser.parse_args()
    teacherLM = args.teacherLM
    model_name = args.studentLM
    analogy_type = args.analogy_type
    dataset = args.dataset
    test_data = read_jsonline(f'dataset/{dataset}/{dataset}_{analogy_type}_{teacherLM}.jsonl')
        
    total_files = len(test_data)
    progress_file = f'progress_QA.txt'
    start_index = get_last_processed_index(progress_file)
    ind = start_index

    with tqdm(total=total_files, desc="Processing files", initial=start_index) as pbar:
        for i, data in enumerate(test_data[start_index:], start=start_index):
            ind = ind + 1
            question = data["question"]["stem"]
            choices = []
            number_mappings = {
                "1": "A",
                "2": "B",
                "3": "C",
                "4": "D",
            }
            if data["answerKey"] in number_mappings:
                data["answerKey"] = number_mappings[data["answerKey"]]
            for ele in data["question"]["choices"]:
                Choice_num = ele["label"]
                if Choice_num in number_mappings:
                    Choice_num = number_mappings[Choice_num]
                Choice_text = ele["text"]
                choices.append(f"{Choice_num}. {Choice_text}")
            
            analogy = data["key_scientific_analogy"]
            answer = answer_generation_analogy(model_name, question, analogy, choices)
            data["answer_analogy"] = {"answer": answer, "analogy": answer_check(answer, data)}

            with open(f'dataset/{dataset}/{dataset}_{analogy_type}_{teacherLM}_{model_name}.jsonl', 'a+', encoding='utf-8') as f:
                line = json.dumps(data, ensure_ascii=False)
                f.write(line + '\n')

            update_progress(progress_file, i + 1)
            pbar.update(1)
            # break


