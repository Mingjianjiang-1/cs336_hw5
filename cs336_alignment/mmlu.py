import json
import os
import re
import csv
from cs336_alignment.utils import load_model, greedy_generate_text, prepare_mmlu_prompts, HOME_DIR


def parse_mmlu_response(response):
    match = re.search(r'The correct answer is ([A-Da-d])', response)
    if match:
        return match.group(1).upper()
    return None

def evaluate_model(llm, examples):
    prompts = prepare_mmlu_prompts(examples)
    outputs = greedy_generate_text(llm, prompts)
    predictions = [parse_mmlu_response(output) for output in outputs]
    
    correct = 0
    total = len(examples)
    failed_to_parse = []
    
    for i, example in enumerate(examples):
        gold_answer = example['answer']
        predicted_answer = predictions[i]
        if predicted_answer is None:
            failed_to_parse.append((example, outputs[i]))
        elif predicted_answer == gold_answer:
            correct += 1
    
    accuracy = correct / total
    return accuracy, failed_to_parse, outputs

def save_results(examples, outputs, accuracy, failed_to_parse, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'accuracy': accuracy,
        'examples': examples,
        'outputs': outputs,
        'failed_to_parse': failed_to_parse
    }
    
    with open(os.path.join(output_dir, 'mmlu_evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
def load_mmlu_examples(filepath, subject):
    # Load examples from a CSV file
    examples = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 6:
                question, A, B, C, D, answer = row
                examples.append({
                    'subject': subject,
                    'question': question,
                    'A': A,
                    'B': B,
                    'C': C,
                    'D': D,
                    'answer': answer
                })
    return examples

if __name__ == "__main__":
    model_path = '/data/Meta-Llama-3-8B'
    # examples_path = 'data/mmlu/test/abstract_algebra_test.csv'
    # examples_path = os.path.join(HOME_DIR, 'data/mmlu/test/abstract_algebra_test.csv')
    # output_dir = 'evaluation_results'
    output_dir = os.path.join(HOME_DIR, 'evaluation_results')
    split_name = 'test'
    
    llm = load_model(model_path)
    # collect all files in data/mmlu/test
    example_paths = os.listdir(os.path.join(HOME_DIR, f'data/mmlu/{split_name}'))
    examples = []
    for example_path in example_paths:
        examples += load_mmlu_examples(os.path.join(HOME_DIR, f'data/mmlu/{split_name}', example_path), example_path.split(f'_{split_name}')[0])
        
    accuracy, failed_to_parse, outputs = evaluate_model(llm, examples)
    
    save_results(examples, outputs, accuracy, failed_to_parse, output_dir)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of failed parses: {len(failed_to_parse)}")
    if len(failed_to_parse) > 0:
        print("Examples of failed parses:")
        for example, output in failed_to_parse[:5]:  # Show first 5 failed parses
            print(f"Example: {example}")
            print(f"Output: {output}")
            print()

    # # Throughput estimation
    # import time
    # start_time = time.time()
    # greedy_generate_text(llm, prompts[:10])  # Generate text for 10 examples to estimate time
    # end_time = time.time()
    # throughput = 10 / (end_time - start_time)
    # print(f"Estimated throughput: {throughput:.2f} examples/second")
