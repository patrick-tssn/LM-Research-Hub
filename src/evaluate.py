import logging
import argparse

from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from transformers import BloomForCausalLM, BloomTokenizerFast
from transformers import OPTForCausalLM, AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM


import csv
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG
)


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Response:"""

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4
)


def init_llama():
    checkpoint = 'output/llama-mix'
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
    model = LlamaForCausalLM.from_pretrained(checkpoint, device_map='auto')
    return tokenizer, model

def init_bloomz():
    checkpoint = 'output/bloomz-mix'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = BloomForCausalLM.from_pretrained(checkpoint, device_map='auto')
    return tokenizer, model

def init_galactica():
    checkpoint = 'output/galactica-mix'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = OPTForCausalLM.from_pretrained(checkpoint, device_map='auto')
    return tokenizer, model

def init_flant5():
    checkpoint = 'output/flant5-mix'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map='auto')
    return tokenizer, model

def init_glm():
    checkpoint = 'output/glm-mix'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map='auto')
    return tokenizer, model

def main(eval_file):

    from collections import defaultdict
    result = defaultdict(list)
    for model_name in ['llama', 'bloomz', 'galactica']:
        if model_name == 'llama':
            tokenizer, model = init_llama()
        elif model_name == 'bloomz':
            tokenizer, model = init_bloomz()
        elif model_name == 'galactica':
            tokenizer, model = init_galactica()
        elif model_name == 'flant5':
            tokenizer, model = init_flant5()
        elif model_name == 'glm':
            tokenizer, model = init_glm()

        print(f'start generate for model {model_name}')

        with open(eval_file, newline='') as ch:
            freader = csv.reader(ch)
            head = next(freader)
            """
            编号, prompt, GPT-3.5评分, GPT3.5, GPT-4评分, GPT-4, 文心一言评分, 文心一言, 参考回答, 任务类型
            """
            for rowlist in tqdm(freader):
                input_text = rowlist[1]
        
                instruction = input_text
                prompt = generate_prompt(instruction)
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].cuda()
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=128
                )
                output = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # print("Response:", output.split("### Response:")[1].strip())
                if model_name == 'flant5':
                    output_text = output.split("### Response:")[0].strip()
                else:    
                    output_text = output.split("### Response:")[1].strip()

                output_text = 'test'
                result[model_name].append(output_text)
    head_index = []
    ori_cases = []
    with open(eval_file, newline='') as ch:
        freader = csv.reader(ch)
        head = next(freader)
        head_index = head[:]
        """
        编号, prompt, GPT-3.5评分, GPT3.5, GPT-4评分, GPT-4, 文心一言评分, 文心一言, 参考回答, 任务类型
        """
        for rowlist in freader:
            # if rowlist is None: continue
            input_text = rowlist[1]
            ori_cases.append(rowlist)

    print('start write to new csv')

    new_cases = []
    for i in range(len(ori_cases)):
        new_case = ori_cases[i]
        for model_name in result.keys():
            print(new_case)
            new_case.insert(-2, result[model_name][i])
        new_cases.append(new_case)
    for model_name in result.keys():
        head_index.insert(-2, model_name)
    
    saved_eval_file = eval_file.replace('.csv', '_new.csv')
    with open('../evaluations/'+saved_eval_file, 'w', newline='') as ch:
        hwriter = csv.writer(ch)
        hwriter.writerow(head_index)
        hwriter.writerows(new_cases)


if __name__ == '__main__':    
    
    main('chinese_eval_basic.csv')
    main('chinese_eval_advanced.csv')
    main('chinese_eval_advanced.csv')

    