import torch
from torch import nn
import os
import numpy as np

from transformers import MllamaForConditionalGeneration, AutoTokenizer, AutoProcessor, TorchAoConfig

from qwen_vl_utils import process_vision_info

from src.utils.clinical_guideline import BIRADS_CLINICAL_GUIDELINE

class Llama:
    def __init__(self, size=3, min_px=256, max_px=1280, freeze=True, device='cuda'):
        model_name = "/model-weights/Llama-3.2-11B-Vision-Instruct/"
        print(os.listdir(model_name))
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(device)

        self.device = device

        if freeze: # Freeze the model
            for param in self.model.parameters():
                param.requires_grad = False
            # Set the model to evaluation mode
            self.model.eval()

        # default processer
        min_pixels = min_px*28*28
        max_pixels = max_px*28*28

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                       min_pixels=min_pixels, 
                                                       max_pixels=max_pixels, 
                                                       pad_token="[PAD]")
        self.processor = AutoProcessor.from_pretrained(model_name, 
                                                       min_pixels=min_pixels, 
                                                       max_pixels=max_pixels, pad_token="[PAD]")

    def run(self, image, metadata, prompt_mode="guidelines"):
        birads = metadata["birads"]

        birads_value = {
            0: 2, 
            1: 3,
            2: 4,
            3: 5,
        }[birads]

        named_concepts = [
            "shadowing",
            "enhancement",
            "halo",
            "calcifications",
            "skin_thickening",
            "circumscribed_margins",
            "spiculated_margins",
            "indistinct_margins",
            "angular_margins",
            "microlobulated_margins",
            "regular_shape",
            "echo_hyperechoic",
            "echo_hypoechoic",
            "echo_heterogeneous",
            "echo_cystic"
        ]
        probabilities = {
            '2': [0.13333333333333333, 0.4, 0.0, 0.03333333333333333, 0.2, 0.8666666666666667, 0.03333333333333333, 0.13333333333333333, 0.0, 0.0, 0.23333333333333334, 0.16666666666666666, 0.3333333333333333, 0.1, 0.03333333333333333],
            '3': [0.08108108108108109, 0.08108108108108109, 0.0, 0.02702702702702703, 0.0, 0.972972972972973, 0.0, 0.02702702702702703, 0.0, 0.0, 0.10810810810810811, 0.0, 0.5405405405405406, 0.1891891891891892, 0.08108108108108109],
            '4': [0.20689655172413793, 0.1206896551724138, 0.27586206896551724, 0.20689655172413793, 0.034482758620689655, 0.05172413793103448, 0.22413793103448276, 0.7931034482758621, 0.20689655172413793, 0.29310344827586204, 0.896551724137931, 0.0, 0.7413793103448276, 0.20689655172413793, 0.05172413793103448],
            '5': [0.625, 0.2, 0.725, 0.15, 0.1, 0.0, 0.375, 0.825, 0.575, 0.25, 1.0, 0.0, 0.8, 0.2, 0.0]
        }

        omit_probabilities_for = [6, 11]

        concept_probabilities = 'Finally, you are given the following probabilities for the presence of certain ultrasound features in this BI-RADS category: \n\n'
        for birads_key in probabilities.keys(): # convert probabilities to percentages
            probabilities[birads_key] = [round(prob, 2) for prob in probabilities[birads_key]]
            for i in range(len(probabilities[birads_key])):
                if i in omit_probabilities_for:
                    continue
                concept_probabilities += f"- {named_concepts[i]}={probabilities[birads_key][i]}"
                if i < len(probabilities[birads_key]) - 1:
                    concept_probabilities += '\n'
        concept_probabilities += '\n\n'

        ask_for_concepts = """
        Please refer to the attached guideline and evaluate the image according to the 15 concepts:
        1. shadowing
        2. enhancement
        3. halo
        4. calcifications
        5. skin_thickening
        6. circumscribed margins
        7. spiculated (uncircumscribed) margins
        8. indistinct (uncircumscribed) margins
        9. angular (uncircumscribed) margins
        10. microlobulated (uncircumscribed) margins
        11. regular shape
        12. echo_hyperechoic
        13. echo_hypoechoic
        14. echo_heterogeneous
        15. echo_cystic

        For each concept, answer with one sentence: 'the image has <concept>, <explanation>.' OR 'the image does not have <concept>, <explanation>.'.
        """

        ask_for_report = """
        Write a report describing the <image> conforming to the provided BI-RADS clinical guidelines.
        """

        ask_for_1_concept = lambda concept: f"""
        You are given the following breast ultrasound <image>. Does the <image> show the following concept:
        - {concept}
        Why or why not? 
        
        Answer with one sentence: 'the image has {concept}. <explanation>' OR 'the image does not have {concept}. <explanation>'.
        """

        prompt_with_mask = f""", along with a segmentation <mask> highlighting a tumor in the image"""

        prompt_with_guidelines = f"""You are given the following breast ultrasound <image>{prompt_with_mask if "mask" in metadata else ""}.
        Its BI-RADS category is {birads_value}.
        You are also given the following clinical guideline: \n\n

        {BIRADS_CLINICAL_GUIDELINE}

        \n\n
        
        {concept_probabilities if prompt_mode == "prob_guidelines" else ""}

        \n\n

        {ask_for_concepts if prompt_mode == "prob_guidelines" else ""}
        {ask_for_report if prompt_mode == "guidelines_noconcepts" else ""}
        """
        
        pathology = {
            0: 'benign tumor',
            1: 'cancer (malignant tumor)',
        }[metadata["y"]]

        labo_prompt = f"""This is a breast ultrasound <image>, labelled as '{pathology}'. Describe what you see in the image. 
        """

        label_free_prompt = f"""This is a breast ultrasound <image>. Perform the following tasks:
        - List the most important features for recognizing this <image> as a '{pathology}' from breast ultrasound.
        - List the things most commonly seen around a '{pathology}' in breast ultrasound.
        - Give superclasses for the word '{pathology}', in the context of breast ultrasound.
        """

        if "guidelines" in prompt_mode:
            prompt = prompt_with_guidelines
        elif prompt_mode == "labo":
            prompt = labo_prompt
        elif prompt_mode == "label_free":
            prompt = label_free_prompt
        elif prompt_mode.startswith("ask_for_"):
            concept_name = prompt_mode.split("ask_for_")[1]
            prompt = ask_for_1_concept(concept_name)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "name": "image"
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        if "mask" in metadata:
            messages[0]["content"].append(
                {
                    "type": "image",
                    "image": metadata["mask"],
                    "name": "mask"
                }
            )

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=768)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

class LLamaGenerator(nn.Module):
    def __init__(self, size=3, min_px=256, max_px=1280, prompt_mode='guidelines', device='cuda'):
        super(LLamaGenerator, self).__init__()
        self.llama = Llama(size=size, min_px=min_px, max_px=max_px, device=device)
        self.device = device
        self.prompt_mode = prompt_mode

    def generate_clinical_report(self, image, metadata):
        if isinstance(image, list):
            if len(image) == 1:
                image = image[0]
                text = self.llama.run(image, metadata, self.prompt_mode)
                return text
            else:
                text = []
                for i in range(len(image)):
                    text.append(self.llama.run(image[i], metadata))
                return text
        else:
            text = self.llama.run(image, metadata, self.prompt_mode)
            return text

    def forward(self, images, birads_list):
        llama_output = []
        for i in range(len(images)):
            image = images[i]
            birads = birads_list[i]
            llama_output.append(
                self.generate_clinical_report(image, birads)
            )
        return llama_output
