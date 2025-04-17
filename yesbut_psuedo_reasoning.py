from datasets import load_dataset
import json

ds = load_dataset("bansalaman18/yesbut")

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


import random
import tqdm
import requests
from PIL import Image
import base64
from io import BytesIO
from PIL import Image
import sys
from pprint import pprint
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--nShot", type=int, default=2)
parser.add_argument("--prompt_mode", required=True, type=str)
parser.add_argument("--model", type=str, default="gpt-4o-mini")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--output_file", required=False)
args = parser.parse_args()


causal_graph_description = """
1. Entities: There is a set of entities (listed in "entities") — each entity can have properties that are either descriptions of/adjectives/adverbs qualifying that entity or (non-causal) relations with other entities. These are listed under “properties” attribute of each entity.
e.g entity: ANIMAL, "properties": ["Cats and Dogs", “pet of HUMAN”, “hides under FURNITURE”], note that HUMAN AND FURNITURE are other entities
entity: FIREWORKS, "properties": [ "Bright and colorful explosions in sky", "burnt by HUMAN”] 
note that the (non—causal) relationships are bidirectional, e.g. FIRECRACKERS (burnt by) HUMAN, HUMAN (burns) FIRECRACKER are same relationships. This relationship is present in "properties" list of any ONE of these entities e.g ("burns FIRECRACKERS" belongs to HUMAN[“properties"]) OR ("burnt by HUMAN" belongs to FIRECRACKER[“properties"]), BUT NOT BOTH 

2. CAUSAL relationships: listed under "causal_relationships". First, we define an EVENT. A collection of entities (along with their (non—causal) relationships) describes an EVENT which is typically of the form "X (optionally) does Y (optionally) with/for/to Z", (a single entity can also be an EVENT) — an EVENT is basically a macro node and a causal relation is defined between events.
A causal relation is listed under "causal_relationships" as a dictionary {"cause": EVENT_1, "effect': EVENT_2}. Each event is expressed in natural language which tells what the collection of entities means, for instance, “X (optionally) does Y (optionally) with/for/to Z”. For example, {"cause": “HUMAN burns FIRECRACKER S”, "effect': “ANIMALS” are frightened}
"""

pseudo_reasoning_labels_nl = "You are given an image and a ground truth label indicating why it is funny/satirical. Based on the exemplars given, explain why this image funny/satirical. To answer this, first create a causal reasoning graph linking different objects, people, and entities present in the image in the form of a piece of code, and then give the final answer. " + "Use the following definition of causal reasoning graph as a guideline: " + causal_graph_description + "Make sure that the final answer is followed after 'FinalAnswerWithoutCode:' "
pseudo_reasoning_labels_vanilla = "You are given an image and a ground truth label indicating why it is funny/satirical. Carefully analyze the entities and the interactions and explain why this image funny/satirical." +"Make sure that the final answer is followed after 'FinalAnswerWithoutCode:' "

prompt_map = {
    "pseudo_reasoning_labels_nl": pseudo_reasoning_labels_nl,
    "pseudo_reasoning_labels_vanilla": pseudo_reasoning_labels_vanilla,
}

stage_data = ds['train']


def pil_image_to_base64(image):
    """Convert PIL image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")



# Function to convert image to base64
def pil_image_to_base64(image):
    """Convert PIL image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

template_file = "yesbut_causal_graph.jsonl"

few_shot_examples = json.load(open(template_file))
few_shot_examples = list(sorted([(k,v) for k,v in few_shot_examples.items()], key=lambda x:x[0]))
few_shot_examples = [f[1] for f in few_shot_examples]
few_shot_examples = [list(f.values()) for f in few_shot_examples]

def getFewShotExamples(nShot):
    try:
        assert nShot <= len(few_shot_examples)
        few_shot_prompt = []
        # print(prompt_map)
        # print(args.prompt_mode)
        for i in range(nShot):
            image_enc = base64.b64encode(requests.get(few_shot_examples[i][0]).content).decode("utf-8")
            #   print(image_enc)
            # print(f"Hi {i}")
            # print(f"prompt map {prompt_map}")
            # print(f"args {args.prompt_mode}")
            few_shot_prompt.append({
                "role": "user",
                "content":  [
                            {"type": "text", "text": prompt_map[args.prompt_mode]},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_enc}",
                                    "detail": "high",
                                },
                            },
                        ],
                    })
            few_shot_prompt.append({
                "role": "assistant",
                "content": f'''{few_shot_examples[i][1]}''' + "\n" + f"Final_Answer_{args.prompt_mode}: {few_shot_examples[i][2]}" if 'vanilla' not in args.prompt_mode.lower()  else f"Final_Answer_{args.prompt_mode}: {few_shot_examples[i][2]}"
            })
            # print("hi")
            # print(f"Few shot examples: {few_shot_prompt}")
    except Exception as e:
        print(f"Error in getFewShotExamples: {e}")
        few_shot_prompt = []
    return few_shot_prompt



from openai import OpenAI

client = OpenAI(api_key='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')

# client = OpenAI(api_key='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')

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def getResponse(image, ground_truth:str, isVanilla: bool, nShot: int):
    try:
        assert nShot in [0,2,5]
        nShotInput = getFewShotExamples(nShot)
        assert len(nShotInput) == nShot*2
        model_inp = nShotInput + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_map[args.prompt_mode]},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{pil_image_to_base64(image)}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": ground_truth},
                    ],
                }
            ]
        
        response = client.chat.completions.create(
            model=args.model,
            messages=model_inp,
            max_tokens=1000,
        )

        assistant_inp = [f for f in model_inp if 'assistant' in f['role']]
        # print(f"Model response: {response.choices[0].message.content, assistant_inp, model_inp[-1]['content'][0]['text']}")
        return response.choices[0].message.content, assistant_inp, model_inp[-1]['content'][0]['text']
    except Exception as e:
        # print(f"Error: {e}")
        return "Error", [], []


import os

def getResponsesAndWriteToFile(nShot:int, debug=False):
  print(f"Generating Response for {nShot} shot Examples with Prompt with mode: {args.prompt_mode}")
  all_random_images = stage_data
  experiment_name = args.prompt_mode
  os.makedirs("yesbut", exist_ok=True)
  if args.output_file:
    outputFileName = args.output_file
  else:
    outputFileName = f"yesbut/yesbut_{experiment_name}_{nShot}_shot_{args.model}.jsonl"
  opened_file = open(outputFileName, "w")
  for i, image in tqdm.tqdm(enumerate(all_random_images)):
      if args.debug:
          if i > 5:
              break
      try:
        #   print(image['overall_description'])
        #   print(image['image'])
        #   print(image['left_image'])
          isvanilla = True if  "vanilla" in args.prompt_mode.lower() else False
          resp, model_inp, prompt = getResponse(image['image'], image["overall_description"], isvanilla, nShot)
          temp = {
            "overall_description": image["overall_description"],
            "stage": image["stage"],
            "difficulty_in_understanding": image["difficulty_in_understanding"],
            "left_image": image["left_image"],
            "right_image": image["right_image"],
            "model": args.model,
            "model_inp": model_inp,
            "response": resp,
            "prompt": prompt,
            "error": ""
            }
      except Exception as e:
          print(e)
          temp = {
                "overall_description": image["overall_description"],
                "stage": image["stage"],
                "difficulty_in_understanding": image["difficulty_in_understanding"],
                "left_image": image["left_image"],
                "right_image": image["right_image"],
                "model": args.model,
                "model_inp": "",
                "prompt": "",
                "response": "Error",
                "error": str(e)
            }
      opened_file.write(json.dumps(temp) + "\n")
      print(f"Data successfully written to {outputFileName}")

if __name__ == '__main__':
    print(f"Number of few shot examples: {args.nShot}")
    getResponsesAndWriteToFile(args.nShot)