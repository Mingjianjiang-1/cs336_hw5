import torch
import torch.nn.functional as F
from cs336_alignment.utils import load_model, HOME_DIR
from cs336_alignment.data_loader import InstructionDataset, get_batches
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset  # assuming the dataset is compatible with Hugging Face datasets
import wandb  # for logging
from data_loader import InstructionDataset, get_batches
import random
import json
import gzip
import argparse
import os

def train(model, tokenizer, train_loader, optimizer, scheduler, device, args):
    model.train()
    total_loss = 0.0
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                print(f"Epoch: {epoch + 1}, Step: {step + 1}, Loss: {total_loss}")
                total_loss = 0.0
                
                # Log training loss
                # wandb.log({"train_loss": total_loss}, step=epoch * len(train_loader) + step)
        
        # # Validate after each epoch
        # validate(model, val_loader, device, epoch)

    # Save the model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

# def validate(model, val_loader, device, epoch):
#     model.eval()
#     total_loss = 0.0
#     with torch.no_grad():
#         for step, batch in enumerate(val_loader):
#             input_ids = batch['input_ids'].to(device)
#             labels = batch['labels'].to(device)
            
#             outputs = model(input_ids, labels=labels)
#             loss = outputs.loss
#             total_loss += loss.item()
    
#     avg_loss = total_loss / len(val_loader)
#     print(f"Epoch: {epoch + 1}, Validation Loss: {avg_loss}")
#     wandb.log({"val_loss": avg_loss}, step=epoch)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3 8B base model")
    parser.add_argument("--output_dir", type=str, default='/home/c-jiangm/spring2024-assignment5-alignment/cs336_alignment/models/sft/', help="Directory to save the trained model and tokenizer")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length for training")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per gradient accumulation step")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Initialize Weights and Biases
    # wandb.init(project="llama-fine-tuning", config=args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = '/data/Meta-Llama-3-8B'
    dataset_path = '/home/shared/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz'
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    ).to(device)
    
    # Load the dataset
    train_dataset = InstructionDataset(tokenizer, dataset_path, args.seq_length, shuffle=True)
    # val_dataset = InstructionDataset(tokenizer, args.dataset_path, args.seq_length, shuffle=False)  # Assuming the same dataset for simplicity; in practice, use a separate validation set
    
    train_loader = get_batches(train_dataset, args.batch_size, shuffle=True)
    # val_loader = get_batches(val_dataset, args.batch_size, shuffle=False)
    
    # Prepare optimizer and schedule (linear warmup and cosine decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.03 * num_training_steps, num_training_steps=num_training_steps)
    
    # Train the model
    train(model, tokenizer, train_loader, optimizer, scheduler, device, args)

if __name__ == "__main__":
    # main()
    # Load the fine-tuned tokenizer and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('/home/c-jiangm/spring2024-assignment5-alignment/cs336_alignment/models/sft/')
    model = AutoModelForCausalLM.from_pretrained('/home/c-jiangm/spring2024-assignment5-alignment/cs336_alignment/models/sft/').to(device)
    
    model.eval()
    
    # Generate text
    prompt = "How do you make a cake?"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(**inputs, max_length=512, num_return_sequences=5, temperature=0.9)