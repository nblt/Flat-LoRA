from data import load_gsm8k
from utils import model_inference, initialize_text_to_text_model
from fire import Fire
import re
import os
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader  # 新增导入
import torch
import os
import typing as tp
import numpy as np
import pandas as pd
from tqdm import tqdm
# from sklearn.metrics import matthews_corrcoef
# from transformers import Seq2SeqTrainer
from transformers import utils
from transformers.utils import import_utils
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import PredictionOutput
from datasets import Dataset, load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from lora_plus import LoraPlusTrainingArguments, LoraPlusTrainer
import logging
import wandb
from peft import PeftModel
from data import load_alpaca

import torch



def model_inference_batch(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    input_text: list,
    model_type: str,
    max_source_length: str = 768,
    max_target_length: str = 256,
):
    if model_type == "CausalLM":
        inputs = tokenizer(
            [text + " " for text in input_text],  
            return_tensors="pt",
            max_length=max_source_length,
            truncation=True,
            padding=True,  
            return_token_type_ids=False,
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=max_target_length,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,  
                # top_p=0.95,
                # temperature=0.8,
                do_sample=False,    
                num_beams=1,        
                top_p=None,
                temperature=None,
            )
        pred_texts = []
        input_ids_len = inputs["input_ids"].shape[1]
        for i in range(len(outputs.sequences)):
            generated = outputs.sequences[i][input_ids_len:]
            pred_text = tokenizer.decode(generated, skip_special_tokens=True)
            pred_texts.append(pred_text)
        
        return pred_texts 


def extract_num(text):
    # Regex pattern to find the number following '####'
    pattern = r'####\s*(\d+)'
    # Using re.search to find the first match
    match = re.search(pattern, text.replace(",", ""))
    if match:
        result = match.group(1)
    else:
        if re.search(r'####\s*-(\d+)', text.replace(",", "")):
            result = re.search(r'####\s*-(\d+)', text.replace(",", "")).group(1)
        else:
            print(text)
            result = ""
    try:
        return int(result.replace(",", ""))
    except:
        print(f"'{result}' can't be converted")
        return 0

def main(lora):


    _, _, test_set = load_gsm8k()
    model_type = "CausalLM"
    model, tokenizer = initialize_text_to_text_model(
        "meta-llama/Llama-2-7b-hf", model_type, True, tokenizer="meta-llama/Llama-2-7b-hf",flash_attention=True
    )
    model = PeftModel.from_pretrained(model, lora)
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", padding_side="left") 
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    all = 0
    correct = 0
    
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
    t = tqdm(test_loader)
    
    wrong_text = ""
    
    for batch in t:
        # breakpoint()
        batch_questions = batch["x"]
        
        
        pred_texts = model_inference_batch(
            model, tokenizer, batch_questions, model_type, max_target_length=512
        )
        

        for x, example, pred_text in zip(batch["x"], batch["y"], pred_texts):
            gt = extract_num(example)
            pred = extract_num(pred_text)
            correct += int(gt == pred)
            all += 1
            if gt!=pred:
                wrong_text += f"Question:\n{x}\nPrediction:\n{pred_text}\nGround Truth:\n{example}\nPrediction Num:{pred}\nGround Truth Num:{gt}\n\n"
                print(f"Question:\n{x}\nPrediction:\n{pred_text}\nGround Truth:\n{example}\nPrediction Num:{pred}\nGround Truth Num:{gt}\n\n")
            
            
        t.set_description(f"Accuracy: {correct/all*100:02f}%")
        
    print("Acc:", correct/all)
    # append to results/gsm8k_results_batch.txt
    if not os.path.exists("results/gsm8k_results_batch.txt"):
        with open("results/gsm8k_results_batch.txt", "w") as f:
            f.write("Model Acc\n")
    with open("results/gsm8k_results_batch.txt", "a") as f:
        f.write(f"{lora} {correct/all}\n")


if __name__ == "__main__":
    Fire(main)
