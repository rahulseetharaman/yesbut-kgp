import torch
import json
import os
from tqdm import tqdm
import random
from datasets import load_dataset
import argparse


from datasets import load_dataset


from openai import OpenAI
import json

ds = load_dataset("bansalaman18/yesbut")

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


import random
import requests
from PIL import Image
import base64
from io import BytesIO
from PIL import Image
import sys
from pprint import pprint
import argparse

# Constants and prompts
SCENE_GRAPH_PROMPT = '''
For the provided image, generate a scene graph in JSON format that includes the following:
1. Objects that are relevant to understanding the humor or satire in this image.
2. Object attributes that contribute to the satirical nature.
3. Object relationships that create the funny context.

Scene Graph:
'''

ANSWER_PROMPT = "Using the image and the scene graph as context, explain why this image is funny or satirical."


# Function to convert image to base64
def pil_image_to_base64(image):
    """Convert PIL image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Helper functions
def load_yesbut_dataset(max_samples=None):
    """Load the full YesBut dataset with satirical images."""
    ds = load_dataset("bansalaman18/yesbut")
    
    # The dataset has multiple stages, collect all satirical images
    satirical_images = []
    
    for stage in ["stage2", "stage3", "stage4"]:
        stage_data = ds['train'].filter(lambda example: example['stage'] == stage)
        for i, image in enumerate(stage_data):
            satirical_images.append({
                "id": f"{stage}-{i}",
                "overall_description": image["overall_description"],
                "stage": image["stage"],
                "difficulty_in_understanding": image["difficulty_in_understanding"],
                "image": image["image"],
                "left_image": image.get("left_image"),
                "right_image": image.get("right_image")
            })
            if max_samples and len(satirical_images) >= max_samples:
                break
        if max_samples and len(satirical_images) >= max_samples:
            break
    
    print(f"Loaded {len(satirical_images)} satirical images from YesBut dataset")
    return satirical_images

def sample_examples(dataset, current_idx, n, difficulty=None):
    """Sample n examples from the dataset, optionally matching difficulty."""
    if n == 0:
        return []
    
    if difficulty:
        # Filter by difficulty if specified
        matching_examples = [
            (i, ex) for i, ex in enumerate(dataset) 
            if i != current_idx and ex['difficulty_in_understanding'] == difficulty
        ]
        
        # If not enough examples match the difficulty, get any examples
        if len(matching_examples) < n:
            other_examples = [
                (i, ex) for i, ex in enumerate(dataset) 
                if i != current_idx and ex['difficulty_in_understanding'] != difficulty
            ]
            samples = matching_examples + random.sample(other_examples, min(n - len(matching_examples), len(other_examples)))
        else:
            samples = random.sample(matching_examples, min(n, len(matching_examples)))
    else:
        # Sample any examples
        candidates = [(i, ex) for i, ex in enumerate(dataset) if i != current_idx]
        samples = random.sample(candidates, min(n, len(candidates)))
    
    # Return only the examples, not the indices
    return [ex for _, ex in samples]

# Load and prepare the model
def load_openai_client():
    model = OpenAI(api_key='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')
    return model

def generate_scene_graph(model, example):
    """Generate a scene graph for an example using MiniCPM.
    Returns the raw scene graph directly from the model (not extracted from final answer).
    """
    # Create the messages for scene graph generation only
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SCENE_GRAPH_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{pil_image_to_base64(example['image'])}",
                        "detail": "high",
                    },
                },
            ]
        }
    ]
    
    # Generate the scene graph in a separate model call, not from the final answer
    response = model.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        max_tokens=256,
    )
    
    scene_graph = response.choices[0].message.content
    
    return scene_graph

def generate_final_answer(model, current_example, scene_graph, context_examples=None):
    """Generate the final answer using the current example and its scene graph."""
    if context_examples is None:
        context_examples = []
        
    messages = []
    
    # Add current example with its scene graph
    graph_prompt = f"Scene Graph: {scene_graph}\n\n{ANSWER_PROMPT}"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": graph_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{pil_image_to_base64(current_example['image'])}",
                        "detail": "high",
                    },
                },
            ]
        }
    ]
    # Generate final answer in a separate model call
    response = model.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        max_tokens=256,
    )
    final_answer = response.choices[0].message.content
    return final_answer

def save_result_to_json(data, filename):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run scene graph analysis on YesBut dataset with MiniCPM-V-2_6")
    parser.add_argument("--cache_dir", type=str, default="/data/user_data/mkapadni/hf_cache/models", 
                        help="Cache directory for models")
    parser.add_argument("--n_shots", type=int, nargs="+", default=[0], 
                        help="Number of examples for few-shot learning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Maximum number of samples to process (for testing)")
    parser.add_argument("--output_dir", type=str, default="yesbut",
                       help="Output directory")
    args = parser.parse_args()
    
    # Set cache directory and seed
    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load the model
    model = load_openai_client()
    
    # Load the dataset
    dataset = load_yesbut_dataset(args.max_samples)
    
    # Run experiments for each N-shot configuration
    for n_shots in args.n_shots:
        print(f"Running {n_shots}-shot experiments with scene graph approach")
        
        # Create output directories
        output_dir = f"{args.output_dir}/{args.seed}/{n_shots}-Shot/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each example
        for i, example in enumerate(tqdm(dataset, desc=f"Processing {n_shots}-shot")):
            try:
                example_id = example["id"]
                
                # First: Generate scene graph directly (not from final answer)
                scene_graph = generate_scene_graph(model, example)
                
                # Save the scene graph to a separate JSON file
                scene_graph_filename = f"yesbut_sg/scene_graphs/gpt4o/{example_id}.json"
                save_result_to_json({"scene_graph": scene_graph}, scene_graph_filename)
                
                # Second: Generate the final answer using the scene graph in a separate call
                final_answer = generate_final_answer(model, example, scene_graph)
                
                # Save the final answer to a separate JSON file
                answer_filename = f"yesbut_sg/final_answers/gpt4o/{example_id}.json"
                save_result_to_json({"final_answer": final_answer}, answer_filename)
                
                # Also save a combined result for convenience
                combined_filename = f"yesbut_sg/combined/gpt4o/{example_id}.json"
                combined_data = {
                    "id": example_id,
                    "stage": example["stage"],
                    "difficulty": example.get("difficulty_in_understanding"),
                    "scene_graph": scene_graph,
                    "final_answer": final_answer,
                    "ground_truth": example["overall_description"]
                }
                save_result_to_json(combined_data, combined_filename)
                
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                
                # Save error information
                error_filename = f"yesbut_sg/errors/gpt4o/{example_id}.json"
                error_data = {
                    "id": example_id,
                    "error": str(e)
                }
                save_result_to_json(error_data, error_filename)
        
        print(f"Completed {n_shots}-shot experiments")

if __name__ == "__main__":
    main()