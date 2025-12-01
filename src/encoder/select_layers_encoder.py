 # This file contains a function adapted from:
 # "call_sequence_classification_model" by Max Klabunde, licensed under CC-BY 4.0.
 # Original source: https://github.com/mklabunde/resi
 # Modified by Keith Ando (2025).


import torch
from transformers import AutoConfig,AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser
from datasets import load_dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def linear_cka(
    features_x: torch.Tensor,
    features_y: torch.Tensor
) -> float:
    """
    Compute linear CKA (centered kernel alignment) between two feature matrices (n x d_x) and (n x d_y).

    """

    if features_x.ndim != 2 or features_y.ndim != 2:
        raise ValueError("features must be 2D (n_examples x dim)")
    features_x = features_x.float()
    features_y = features_y.float()

    x = features_x - features_x.mean(dim=0, keepdim=True)
    y = features_y - features_y.mean(dim=0, keepdim=True)

    k_xy = x.t().mm(y)   
    k_xx = x.t().mm(x)   
    k_yy = y.t().mm(y)   

    dot_xy = torch.norm(k_xy).pow(2) 
    norm_x = torch.norm(k_xx)
    norm_y = torch.norm(k_yy)

    denom = (norm_x * norm_y).clamp(1e-12)
    cka_val = (dot_xy / denom).item()
    return float(cka_val)

@dataclass
class Arguments:
    model_name_or_path: str = field()
    glue_task: str = field()
    number_of_samples: Optional[int] = field(default=None)
    print_layers: Optional[bool] = field(default=False)
    batch_size: int = field(default=128)

def extract_from_sequence_classification_model(
        model,
        tokenizer,
        prompt,
        device,
        max_length,
        with_text_pair = False,
        padding_type = "max_length",
        batch_size = 1,
        layers_to_extract = [-1],
        feature_selection = "cls_token"
) -> Tuple[torch.Tensor,...]:
    all_features = {layer: [] for layer in layers_to_extract}

    if with_text_pair == True:
        # Expects prompt to be a list of lists with the pair of texts like [[text1,text2],[text1,text2]...]
        for i in range(0, len(prompt),batch_size):
            batch_prompts = prompt[i:i+batch_size]
            model_inputs = tokenizer(text = [text[0] for text in batch_prompts],
                                    text_pair = [text[1] for text in batch_prompts], 
                                    return_tensors="pt",
                                    max_length = max_length,
                                    padding = padding_type,
                                    truncation = True).to(device)
            input_ids = model_inputs["input_ids"].to(device)
#            token_type_ids = model_inputs["token_type_ids"].to(device) 
            attention_mask = model_inputs["attention_mask"].to(device)  
            with torch.no_grad():
                model_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            for layer in layers_to_extract:
                if feature_selection == "cls_token":
                    activations = model_outputs.hidden_states[layer][:, 0, :] # First token, on Roberta and Bert, Classification Token
                elif feature_selection == "mean":
                    activations = model_outputs.hidden_states[layer]  #(batch, seq_len, model_dim)

                    unsqueezed_attention_mask = attention_mask.unsqueeze(-1)  #(batch, seq_len, 1)

                    # Compute mean excluding padding
                    sum_activations = (activations * unsqueezed_attention_mask).sum(dim=1)  
                    valid_token_counts = unsqueezed_attention_mask.sum(dim=1)  
                    activations = sum_activations / valid_token_counts  
                all_features[layer].append(activations.cpu()) 
    else:
        for i in range(0, len(prompt),batch_size):
            batch_prompts = prompt[i:i+batch_size]
            model_inputs = tokenizer(batch_prompts, 
                                    return_tensors="pt",
                                    max_length = max_length,
                                    padding = padding_type,
                                    truncation = True).to(device)
        
            input_ids = model_inputs["input_ids"].to(device)
            attention_mask = model_inputs["attention_mask"].to(device)  
            with torch.no_grad():
                model_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            for layer in layers_to_extract:
                if feature_selection == "cls_token":
                    activations = model_outputs.hidden_states[layer][:, 0, :] # First token, on Roberta and Bert, Classification Token
                elif feature_selection == "mean":
                    activations = model_outputs.hidden_states[layer]  #(batch, seq_len, model_dim)

                    unsqueezed_attention_mask = attention_mask.unsqueeze(-1)  #(batch, seq_len, 1)

                    # Compute mean excluding padding
                    sum_activations = (activations * unsqueezed_attention_mask).sum(dim=1)  
                    valid_token_counts = unsqueezed_attention_mask.sum(dim=1)  
                    activations = sum_activations / valid_token_counts  
                all_features[layer].append(activations.cpu()) 

    # Making it just a tensor n examples x d model_dimension
    for layer in layers_to_extract:
        all_features[layer] = torch.cat(all_features[layer], dim=0)

    return all_features

def main():
    parser = HfArgumentParser(Arguments)

    args = parser.parse_args_into_dataclasses()[0]
    dataset = load_dataset("nyu-mll/glue", args.glue_task)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
         
    start = time.time()
    if args.number_of_samples is not None:
        dataset["train"] = dataset["train"].select(range(args.number_of_samples))
    

    sentence1_key, sentence2_key = task_to_keys[args.glue_task]


    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:

        base = getattr(model, getattr(model, "base_model_prefix", ""), model)
        if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
            num_layers = len(base.encoder.layer)
        elif hasattr(base, "encoder") and hasattr(base.encoder, "layers"):
            num_layers = len(base.encoder.layers)
        elif hasattr(base, "transformer") and hasattr(base.transformer, "h"):
            num_layers = len(base.transformer.h)
        else:
            raise RuntimeError("Could not determine number of transformer layers from the model.")

    layers_to_extract = list(range(0, num_layers + 1))
    print(f"Detected num_hidden_layers={num_layers}, layers_to_extract={layers_to_extract}")


    if sentence2_key == None:
        features_map = extract_from_sequence_classification_model(model,tokenizer,[element for element in dataset["train"][sentence1_key]],device = "cuda", max_length = 512,batch_size=args.batch_size,layers_to_extract=layers_to_extract)
    else:
        features_map = extract_from_sequence_classification_model(model,tokenizer,[[element[sentence1_key],element[sentence2_key]] for element in dataset["train"]],device = "cuda", max_length = 512,batch_size=args.batch_size,layers_to_extract=layers_to_extract,with_text_pair = True)
    
    cka_list = []
    
    for i in range(0,len(layers_to_extract) - 1):
        cka_list.append((linear_cka(features_map[i],features_map[i+1]),i))
    end = time.time()
    print(f"Total time: {end - start}s")
 
    if args.print_layers:
        for cka_value, layer in cka_list:
            print(f"Layer {layer} to {layer + 1}: CKA = {cka_value}")
    
    import json
    cka_dict = {f"layer_{layer}_to_{layer + 1}": cka_value for cka_value, layer in cka_list}
    with open(f"cka_layers_{args.glue_task}.json", "w") as f:
        json.dump(cka_dict, f, indent=4)

    cka_list_sorted = sorted(cka_list, key=lambda x: x[0])
    print(cka_list_sorted)

    one_hot = [1 if i in [layer for _, layer in cka_list_sorted[1:int(len(cka_list_sorted)*0.5 + 1)]] else 0 for i in range(len(cka_list_sorted))]
    with open(f"cka_layers_{args.glue_task}_onehot.json", "w") as f:
        json.dump(one_hot, f, indent=4)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()