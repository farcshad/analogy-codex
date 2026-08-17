# — coding: utf-8 –
import json
import argparse
from util_func import *
import os
import json
from tqdm import tqdm


def sci_concept_generation(question):
    prompt = f'''
Given a scientific question, you should show the key scientific concept related to this scientific question.
This is a scientific question:
{question}
You should only output in a parsible JSON format. The example outputs look like:\n
{{\"key_scientific_concept\": \"The_key_scientific_concept\"}}\n
The key scientific concept:
'''
    messages = [
            {"role": "user", "content": prompt},
        ]
    concept = evaluator_construction(messages, 'gpt-4o', data_type=os.environ["DATA_TYPE"])
    return concept


def free_form_analogy(model_name, concept):
    prompt = f"Please use an analogy with no more than 300 words to explain the scientific concept: {concept}\nAnalogy"
    messages = [
            {"role": "user", "content": prompt},
        ]
    analogy = evaluator_construction(messages, model_name, data_type=os.environ["DATA_TYPE"])
    return analogy

def word_analogy(model_name, concept):
    prompt = f'''
Given one scientific concept, you should use another concept as an analogy to explain this concept. 
This is an example:
The scientific concept: Microclimate\nAnalogy: Microclimate can be analogous to a Classroom Environment
You should output the analogy of similar form to the example.
The scientific concept: {concept}
Analogy:
'''
    messages = [
            {"role": "user", "content": prompt},
        ]
    analogy = evaluator_construction(messages, model_name, data_type=os.environ["DATA_TYPE"])
    return analogy

def structure_analogy(model_name, concept):
    prompt = f'''
Given one scientific concept, you should use another concept as an analogy to explain this concept. Moreover, you should use other concepts that are related to these two concept to explain the analogy.\n
This is an example:
The scientific concept: Microclimate\nAnalogy: Microclimate can be analogous to a Classroom Environment\n\n1. Temperature and humidity variation correspond to classroom temperature and noise levels:\nIn a microclimate, small-scale variations in temperature and humidity can greatly affect the living conditions for plants, animals, and insects. Similarly, in a classroom, the temperature and noise levels can significantly influence the comfort and concentration of students. In both cases, these environmental factors play a crucial role in the well-being and activity level of the inhabitants.\n\n2. Local vegetation corresponds to educational decorations and tools:\nJust as the type and density of vegetation can create and define a microclimate by providing shade or retaining moisture, educational decorations (such as posters and models) and tools (like whiteboards and projectors) shape the learning environment of a classroom. They both serve to enhance the experience of those within the space, either by providing shelter and resources or by facilitating learning and engagement.\n\n3. Topography corresponds to classroom layout:\nThe shape and features of the land (topography) influence the formation of microclimates by directing wind patterns or affecting sun exposure. Analogously, the layout of a classroom, including the arrangement of desks and the presence of distinct areas for different activities, can influence how students interact, move, and focus. The physical structure in both cases dictates the flow and distribution of the inhabitants' activities.\n\n4. Human-made structures correspond to classroom rules and routines:\nIn a microclimate, human-made structures such as buildings, roads, and walls can alter the local climate by affecting airflow, sunlight, and temperature. Similarly, in a classroom, established rules and routines create a structured environment that dictates the behavior and interactions of students. Both human-made structures and classroom rules and routines impose an artificial order that can either enhance or disrupt the natural dynamics of the system.\n\nThese two concepts can be analogous because they both represent localized environments that are influenced by a combination of physical factors and living or artificial elements. These environments have their own distinct characteristics that can vary significantly from the surrounding areas, and they both require careful management and understanding to maintain their optimal conditions.\n
You should output the analogy of similar form to the example.
The scientific concept: {concept}
Analogy:
'''
    messages = [
            {"role": "user", "content": prompt},
        ]
    analogy = evaluator_construction(messages, model_name, data_type=os.environ["DATA_TYPE"])
    return analogy

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4-0613")
    parser.add_argument("--analogy_type", type=str, default="free-form", help='free-form, structure and word')
    parser.add_argument("--dataset", type=str, default="ARC", help='ARC or GPQA')
    args = parser.parse_args()
    model_name = args.model_name
    analogy_type = args.analogy_type
    dataset = args.dataset
    test_data = read_jsonline(f'dataset/{dataset}/{dataset}_dataset.jsonl')    
    total_files = len(test_data)
    progress_file = f'progress_generation.txt'
    start_index = get_last_processed_index(progress_file)
    ind = start_index
    with tqdm(total=total_files, desc="Processing files", initial=start_index) as pbar:
        for i, data in enumerate(test_data[start_index:], start=start_index):
            ind = ind+1
            question = data["question"]["stem"]
            choices = []
            number_mappings = {
                "1": "A",
                "2": "B",
                "3": "C",
                "4": "D",
            }
            for ele in data["question"]["choices"]:
                Choice_num = ele["label"]
                if Choice_num in number_mappings:
                    Choice_num = number_mappings[Choice_num]
                Choice_text = ele["text"]
                choices.append(f"{Choice_num}: {Choice_text}")
            
            concept = sci_concept_generation(question)
            data["key_scientific_concept"] = concept
            
            if analogy_type == 'free-form':
                analogy = free_form_analogy(model_name, concept)
            if analogy_type == 'structure':
                analogy = structure_analogy(model_name, concept)
            if analogy_type == 'word':
                analogy = word_analogy(model_name, concept)
            data["key_scientific_analogy"] = analogy

            with open(f'dataset/{dataset}/{dataset}_{analogy_type}_{model_name}.jsonl', 'a+', encoding='utf-8') as f:
                line = json.dumps(data, ensure_ascii=False)
                f.write(line + '\n')

            update_progress(progress_file, i + 1)
            pbar.update(1)