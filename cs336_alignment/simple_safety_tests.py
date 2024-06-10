import csv
import json
import os
import time
from cs336_alignment.utils import load_model, greedy_generate_text, HOME_DIR, prepare_alpacaeval_prompts

def load_simplesafetytests_examples(filepath):
    examples = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append({
                'id': row['id'],
                'harm_area': row['harm_area'],
                'counter': row['counter'],
                'category': row['category'],
                'instruction': row['prompts_final']
            })
    return examples

def generate_outputs(llm, examples):
    results = []
    
    prompts = prepare_alpacaeval_prompts(examples)
   
    batch_outputs = greedy_generate_text(llm, prompts=prompts)
    
    for j, output in enumerate(batch_outputs):
        instruction = examples[j]['instruction']
        result = {
            'prompts_final': instruction,
            'output': output
        }
        results.append(result)

    return results

def save_results(results, output_path):
    with open(output_path, 'w') as fout:
        for result in results:
            fout.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    model_path = '/data/Meta-Llama-3-8B'
    dataset = 'simple_safety_tests'
    examples_path = os.path.join(HOME_DIR, f'data/{dataset}/{dataset}.csv')
    output_path = os.path.join(HOME_DIR, 'evaluation_results', f'{dataset}_output.json')
    
    llm = load_model(model_path)
    examples = load_simplesafetytests_examples(examples_path)
    
    start_time = time.time()
    results = generate_outputs(llm, examples)
    end_time = time.time()
    
    save_results(results, output_path)
    
    # total_time = end_time - start_time
    # throughput = len(examples) / total_time
    # print(f"Estimated throughput: {throughput:.2f} examples/second")
