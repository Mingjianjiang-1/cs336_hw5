from vllm import LLM, SamplingParams
import re

HOME_DIR = '/home/c-jiangm/spring2024-assignment5-alignment/'

def load_model(model_path):
    return LLM(model=model_path)

def generate_text(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def greedy_generate_text(llm, prompts):
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    )
    outputs = llm.generate(prompts, sampling_params)
    
    output_texts = []
    for output in outputs:
        output_texts.append(output.outputs[0].text)

    return output_texts

# def parse_mmlu_responses(response):
#     # Parse the response from the Meta-Llama API. The response is a list of strings.
#     output_texts = []
#     for output in response:
#         parsed_text = re.sub('[^a-zA-Z]', '', output).lower()
#         output_texts.append(parsed_text)
#     return output_texts

def load_system_prompt():
    with open('prompts/zero_shot_system_prompt.prompt', 'r') as f:
        system_prompt = f.read()
    return system_prompt
    
def prepare_mmlu_prompts(batch):
    system_prompt = load_system_prompt()
    with open('prompts/mmlu.prompt', 'r') as f:
        mmlu_template = f.read()
        
    prompts = []
    for i in range(len(batch)):
        example = batch[i]
        mmlu_prompt = mmlu_template.format(subject=example['subject'], question=example['question'], A=example['A'], B=example['B'], C=example['C'], D=example['D'])
        prompt = system_prompt.format(instruction=mmlu_prompt)
        prompts.append(prompt)
    return prompts

def prepare_gsm8k_prompts(batch):
    system_prompt = load_system_prompt()
    with open('prompts/gsm8k.prompt', 'r') as f:
        gsm8k_template = f.read()
        
    prompts = []
    for i in range(len(batch)):
        example = batch[i]
        gsm8k_prompt = gsm8k_template.format(question=example['question'])
        prompt = system_prompt.format(instruction=gsm8k_prompt)
        prompts.append(prompt)
    return prompts

def prepare_alpacaeval_prompts(batch):
    system_prompt = load_system_prompt()
    
    prompts = []
    for i in range(len(batch)):
        example = batch[i]
        prompt = system_prompt.format(instruction=example['instruction'])
        prompts.append(prompt)
    return prompts

def prepare_sst_prompts(batch):
    system_prompt = load_system_prompt()
    
    prompts = []
    for i in range(len(batch)):
        example = batch[i]
        prompt = system_prompt.format(instruction=example['prompts_final'])
        prompts.append(prompt)
    return prompts
                                      
if __name__ == '__main__':
    print(load_system_prompt())
    
    
    
    # # Sample prompts.
    # prompts_ = [
    #     "Hello, my name is",
    #     "The president of the United States is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ]
 
    # # Load the model.
    # llm = load_model('/data/Meta-Llama-3-8B')
    # output_texts = greedy_generate_text(llm, prompts_)
    # print(output_texts) 