
from cs336_alignment.utils import load_model, greedy_generate_text, prepare_mmlu_prompts, HOME_DIR
import json
import gzip
import random
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
import os
import torch

class InstructionDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, dataset_path: str, seq_length: int, shuffle: bool = True):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.data = []
        self.token_lengths = []
        # Make dataset_path a PosixPath object
        dataset_path = os.fspath(dataset_path)
        
        # If the dataset is a .jsonl file, load it as a list of dictionaries
        if dataset_path.endswith('.jsonl'):
            with open(dataset_path, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))
        elif dataset_path.endswith('.jsonl.gz'):
            with gzip.open(dataset_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line))
        self.data = self.data[:100]
        print('EOT token:', self.tokenizer.eos_token)
        print('EOT token ID:', self.tokenizer.eos_token_id)
        print('BOS token:', self.tokenizer.bos_token)
        
        prompt_path = os.path.join(HOME_DIR, 'cs336_alignment/prompts/alpaca_sft.prompt')
        with open(prompt_path, "r") as f:
            self.alpaca_template = f.read()

        random.seed(42)
        if self.shuffle:
            random.shuffle(self.data)
        
        for i, item in enumerate(self.data):
            prompt = item['prompt']
            response = item['response']
            formatted_text = self.alpaca_template.format(instruction=prompt, response=response)
            tokenized_text = self.tokenizer(self.tokenizer.bos_token + formatted_text + self.tokenizer.eos_token, return_tensors='pt', add_special_tokens=False)
            self.token_lengths.append(len(tokenized_text['input_ids'][0]))

        self.total_length = sum(self.token_lengths)
        self.num_sequences = self.total_length // self.seq_length 
        print('Total length:', self.total_length)
        print('Number of sequences:', self.num_sequences)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        if end_idx >= self.total_length:
            raise IndexError('Index out of range')

        current_length = 0
        token_sequences = []
        labels_sequences = []

        for i, length in enumerate(self.token_lengths):
            if current_length + length > start_idx:
                # Calculate the starting point within the current text
                token_start_idx = max(0, start_idx - current_length)
                token_end_idx = min(length, end_idx - current_length)
                
                prompt = self.data[i]['prompt']
                response = self.data[i]['response']
                formatted_text = self.alpaca_template.format(instruction=prompt, response=response)
                tokenized_text = self.tokenizer(self.tokenizer.bos_token + formatted_text + self.tokenizer.eos_token, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
                token_sequences.append(tokenized_text[token_start_idx:token_end_idx])
                labels_sequence = tokenized_text[token_start_idx+1:token_end_idx+1]
                if token_end_idx == length:
                    labels_sequence = torch.cat((labels_sequence, torch.tensor([self.tokenizer.bos_token_id])))
                labels_sequences.append(labels_sequence)
            
                start_idx = current_length + token_end_idx
                if start_idx >= end_idx:
                    break

            current_length += length

        input_ids = torch.cat(token_sequences)
        labels = torch.cat(labels_sequences)

        return {
            'input_ids': input_ids,
            'labels': labels
        }

### 2. Function to Return Batches from the Dataset

def get_batches(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Testing the implementation
if __name__ == "__main__":
    llm = load_model('/data/Meta-Llama-3-8B')
    tokenizer = llm.get_tokenizer()
    dataset_path = '/home/shared/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz'
    seq_length = 128
    
    # dataset = InstructionDataset(tokenizer, dataset_path, seq_length)

    # print('Number of sequences:', len(dataset))
    # print('First sequence:', dataset[0])
    # print('Second sequence:', dataset[1])
    
    # batch_size = 2
    # batches = get_batches(dataset, batch_size)

