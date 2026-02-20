import torch
import torch.nn as nn
import numpy as np

from src.utils.clinical_guideline import (
    BIRADS_DIAGNOSTIC_GUIDELINE_US, 
    BIRADS_DIAGNOSTIC_GUIDELINE_MG, 
    BIRDS_FIELD_GUIDE
)

from src.utils.dataloaders import named_concept_bank
from src.utils.classes import CUB


from transformers import AutoModelForCausalLM, AutoTokenizer
import json

import re


def normalize_species_name(raw_name: str) -> str:
    """
    Convert species names from classes.txt style (with leading numbers/underscores)
    into dictionary keys used in birds_guideline.
    
    Example:
        "001.Black_footed_Albatross" -> "Black-footed Albatross"
        "121.White_crowned_Sparrow"  -> "White-crowned Sparrow"
    """
    # Remove any leading numbers and dot (e.g., "001.")
    name = re.sub(r"^\d+\.", "", raw_name)
    
    # Replace underscores with spaces
    name = name.replace("_", " ")
    
    # Hyphenate common compound modifiers
    name = re.sub(
        r"\b([A-Z][a-z]+) (footed|winged|capped|crowned|necked|throated|tailed|backed)\b",
        r"\1-\2",
        name,
    )
    
    # Capitalize words if they appear lowercased
    name = " ".join([w.capitalize() if w.islower() else w for w in name.split()])
    
    return name


class Qwen3:
    def __init__(self, device='cuda'):        
        model_name = f"/model-weights/Qwen3-8B"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        ).to(device)
        
        with open("/model-weights/Qwen3-8B/tokenizer_config.json") as f:
            tokenizer_config = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        self.device = device

    def run(self, metadata):
        concepts = metadata['concepts']
        y = metadata['y']
        y_pred = metadata['y_pred']
        dataset_name = metadata["dataset"]
        named_concepts = named_concept_bank(dataset=dataset_name)
        selected_concepts = metadata["selected_concepts"]
        species_name = metadata["species_name"]

        type_of_guideline = {
            "BUSBRA": "clinical",
            "BREAST_US": "clinical",
            "BrEaST": "clinical",
            "DDSM": "clinical",
            "CUB": "bird watching"
        }[dataset_name]

        introduction = ''
        if dataset_name == "BREAST_US" or dataset_name == "DDSM":
            diagnosis = {0: "Benign", 1: "Cancer"}[y_pred]
            introduction = f"""You are given the final diagnostic prediction of a AI model for breast cancer diagnosis, which is {diagnosis}. 
            Benign predictions can range from to BI-RADS 2 to BI-RADS 4, while malignant predictions correspond to BI-RADS 4A–5, reflecting different levels of suspicion for malignancy.
            The model has also detected the following concepts in the image: """
        
        if dataset_name == "CUB":
            y_pred = int(y_pred.cpu().item())
            species_name = CUB.class_names[y_pred + 1]
            species_name = normalize_species_name(species_name)
            introduction = f"""You are given the final prediction of a AI model for bird species identification, which is {species_name}. 
            The model has also detected the following concepts in the image: """

        GUIDELINE = {
            "BUSBRA": BIRADS_DIAGNOSTIC_GUIDELINE_US,
            "BrEaST": BIRADS_DIAGNOSTIC_GUIDELINE_US,
            "BREAST_US": BIRADS_DIAGNOSTIC_GUIDELINE_US,
            "DDSM": BIRADS_DIAGNOSTIC_GUIDELINE_MG,
            "CUB": BIRDS_FIELD_GUIDE[species_name] if species_name else ''
        }[dataset_name]

        # Prepare the concept data for the prompt
        concept_data = ''
        for i in range(len(selected_concepts)):
            c_pred_i = metadata["concepts"][i] * 100
            if metadata["concepts"][i] >= 0.5:
                concept_data += f"{named_concepts[i].capitalize().replace('_', ' ')} ({c_pred_i}% confidence)\n"
            else:
                if dataset_name == "BREAST_US" and named_concepts[i] == 'regular_shape':
                    concept_data += f"Irregular shape ({c_pred_i}% confidence)\n"

        instructions = ''
        if dataset_name in ["BREAST_US", "DDSM"]:
            instructions += f"""Assuming the diagnosis is correct, what are the implications of these concepts for the final diagnosis based on the BIRADS clinical guideline, provided below?
            Based on the guideline, explain the predicted concepts, analyze how those concepts align with the models' prediction, determine the most likely BI-RADS category and give follow-up recommendation. """
        if dataset_name == "CUB":
            instructions += f"""Assuming the final prediction is correct, what are the implications of these concepts for the final species classification based on an excerpt from the bird watching field guide, provided below?
            Based on the guideline, explain the predicted concepts, and analyze how those concepts align with the models' prediction. """
        
        reasoning_prompt = f"""{introduction}

        {concept_data}
        
        {instructions}

        {GUIDELINE}
        """

        self.prompt = reasoning_prompt

        messages = [
            {"role": "user", "content": reasoning_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switch between thinking and non-thinking modes. Default is True.
        )
       
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return thinking_content, content

    def get_prompt(self):
        return self.prompt
    

class ReasoningModel(nn.Module):
    def __init__(self, device='cuda'):
        super(ReasoningModel, self).__init__()
        self.model = Qwen3(device=device)
        self.device = device

    def generate_reasoning(self, metadata):
        text = self.model.run(metadata)
        return text

    def forward(self, metadata):
        return self.generate_reasoning(metadata)

    def get_prompt(self):
        return self.model.get_prompt()
