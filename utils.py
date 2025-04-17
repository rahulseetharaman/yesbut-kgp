import requests
from PIL import Image
import json
from pprint import pprint
from io import BytesIO
import base64
from openai import OpenAI
from tqdm import tqdm
import time
import argparse
import os

def read_image_from_url(url, show=False):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes

        # Open the image from the byte content
        img = Image.open(BytesIO(response.content))

        if show:
            img.show()

        return img

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the image: {e}")
        return None
    except IOError:
        print("Error opening the image.")
        return None


def extract_meme_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()

        extracted_data = []

        for i, entry in enumerate(data):
            title = entry.get("title", "")
            meme_url = entry.get("url", "")
            meme_captions = entry.get("meme_captions", [])

            # Check if url and meme_captions are non-empty
            if meme_url and meme_captions:
                extracted_data.append({
                    "title": title,
                    "url": meme_url,
                    "meme_captions": meme_captions
                })
            else:
                print(f"Skipping entry {i+1}: Missing URL or meme_captions.")

        return extracted_data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the JSON file: {e}")
        return []
    except json.JSONDecodeError:
        print("Error decoding JSON.")
        return []

def pil_image_to_base64(image):
    """
    Converts a PIL Image to a base64 encoded string.
    """
    buffered = BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_response(client, image, prompt, model_name, fewshot_templates, retries=1, delay=5):
    """
    Sends an image and prompt to GPT, retries on failure.

    Parameters:
    image: PIL Image object.
    prompt: Prompt text to send along with the image.
    retries: Number of retries in case of failure.
    delay: Delay between retries in seconds.

    Returns:
    GPT response text if successful, None otherwise.
    """
    base64_image = pil_image_to_base64(image)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=fewshot_templates + [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    # "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000,
            )
            return response.choices[0].message.content

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return f"<<ERROR>> {str(e)}"