import json
import os
import time
import re
from cs336_alignment.utils import load_model, generate_text, greedy_generate_text, HOME_DIR, prepare_gsm8k_prompts

def parse_gsm8k_response(response):
    matches = re.findall(r'(\d+)', response)
    if matches:
        return matches[-1]
    return None

def evaluate_model(llm, examples):
    prompts = prepare_gsm8k_prompts(examples)
    outputs = greedy_generate_text(llm, prompts)
    predictions = [parse_gsm8k_response(output) for output in outputs]
    
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
    
    with open(os.path.join(output_dir, 'gsm8k_evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
def load_gsm8k_examples(filepath):
    examples = []
    with open(filepath, 'r') as f:
        for line in f:
            example = json.loads(line.strip())
            question = example['question']
            answer_str = example['answer'].split('#### ')[-1]
            answer = int(answer_str.replace(',', ''))
            examples.append({
                'question': question,
                'answer': answer
            })
    return examples

if __name__ == "__main__":
    model_path = '/data/Meta-Llama-3-8B'
    examples_path = os.path.join(HOME_DIR, 'data/gsm8k/test.jsonl')
    output_dir = os.path.join(HOME_DIR, 'evaluation_results')
    
    llm = load_model(model_path)
    examples = load_gsm8k_examples(examples_path)
    
    start_time = time.time()
    accuracy, failed_to_parse, outputs = evaluate_model(llm, examples)
    end_time = time.time()
    
    save_results(examples, outputs, accuracy, failed_to_parse, output_dir)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of failed parses: {len(failed_to_parse)}")
    if len(failed_to_parse) > 0:
        print("Examples of failed parses:")
        for example, output in failed_to_parse[:5]:  # Show first 5 failed parses
            print(f"Example: {example}")
            print(f"Output: {output}")
            print()

    # Throughput estimation
    total_time = end_time - start_time
    throughput = len(examples) / total_time
    print(f"Estimated throughput: {throughput:.2f} examples/second")
