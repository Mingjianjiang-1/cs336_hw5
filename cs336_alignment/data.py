import os
import json
import gzip
import random
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from cs336_alignment.utils import load_model, greedy_generate_text, prepare_mmlu_prompts, HOME_DIR


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
        
        print('EOT token:', self.tokenizer.eos_token)
        print('EOT token ID:', self.tokenizer.eos_token_id)
        
        prompt_path = os.path.join(HOME_DIR, 'cs336_alignment/prompts/alpaca_sft.prompt')
        with open(prompt_path, "r") as f:
            self.alpaca_template = f.read()

        random.seed(42)
        if self.shuffle:
            random.shuffle(self.data)

        for item in self.data:
            prompt = item['prompt']
            response = item['response']
            formatted_text = self.alpaca_template.format(instruction=prompt, response=response)
            tokenized_text = self.tokenizer(formatted_text + self.tokenizer.eos_token, return_tensors='pt', add_special_tokens=False)
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
        concatenated_text = ""

        for i, length in enumerate(self.token_lengths):
            if current_length + length > start_idx:
                # Calculate the starting point within the current text
                token_start_idx = max(0, start_idx - current_length)
                token_end_idx = min(length, end_idx - current_length)
                
                prompt = self.data[i]['prompt']
                response = self.data[i]['response']
                formatted_text = self.alpaca_template.format(instruction=prompt, response=response)
                # tokenized_text = self.tokenizer(formatted_text + self.tokenizer.eos_token, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
                # concatenated_text += self.tokenizer.decode(tokenized_text[token_start_idx:token_end_idx])
                concatenated_text += formatted_text + self.tokenizer.eos_token
                
                if current_length + token_end_idx >= end_idx:
                    break

            current_length += length

        final_tokenized_text = self.tokenizer(concatenated_text, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
        
        input_ids = final_tokenized_text[:self.seq_length]
        labels = final_tokenized_text[1:self.seq_length+1]

        # Padding if necessary
        if len(labels) < self.seq_length:
            padding_length = self.seq_length - len(labels)
            labels = torch.cat([labels, torch.full((padding_length,), -100)])
            input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])

        return {
            'input_ids': input_ids,
            'labels': labels
        }
