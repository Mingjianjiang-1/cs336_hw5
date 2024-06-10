import json
import os
import time
from cs336_alignment.utils import load_model, greedy_generate_text, prepare_alpacaeval_prompts, HOME_DIR

def load_alpacaeval_examples(filepath):
    examples = []
    with open(filepath, 'r') as f:
        for line in f:
            example = json.loads(line.strip())
            examples.append(example)
    return examples

def generate_outputs(llm, examples):
    prompts = prepare_alpacaeval_prompts(examples)
    results = []
    
    output = greedy_generate_text(llm, prompts=prompts)
        
    for j, output in enumerate(output):
        instruction = examples[j]['instruction']
        dataset = examples[j]['dataset']
        result = {
            'instruction': instruction,
            'output': output,
            'generator': 'llama-3-8b-base',
            'dataset': dataset
        }
        results.append(result)

    return results

def save_results(results, output_path):
    with open(output_path, 'w') as fout:
        json.dump(results, fout, indent=4)

if __name__ == "__main__":
    model_path = '/data/Meta-Llama-3-8B'
    dataset = 'alpaca_eval'
    examples_path = os.path.join(HOME_DIR, f'data/{dataset}/{dataset}.jsonl')
    output_path = os.path.join(HOME_DIR, 'evaluation_results', f'{dataset}_output.json')
    
    llm = load_model(model_path)
    examples = load_alpacaeval_examples(examples_path)
    
    start_time = time.time()
    results = generate_outputs(llm, examples)
    end_time = time.time()
    
    save_results(results, output_path)
    
    # total_time = end_time - start_time
    # throughput = len(examples) / total_time
    # print(f"Estimated throughput: {throughput:.2f} examples/second")
