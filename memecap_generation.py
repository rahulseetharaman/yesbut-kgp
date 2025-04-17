
from utils import *

parser=argparse.ArgumentParser()
parser.add_argument('--dataset_name', required=True, help='Name of the dataset')
parser.add_argument('--output_dir', required=True, help='Output directory')
parser.add_argument('--model_name', required=True, help='Model name')
parser.add_argument('--mode', required=True, help='Choose between causal and vanilla')
parser.add_argument('--nshot', required=True, help='Number of shots for few-shot learning', type=int)
parser.add_argument('--key', required=True, help='Alternate API Key')
parser.add_argument('--base_url', required=True, help='Alternate Base URL')
OPENAI_API_KEY = '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'


def save_to_json(file_path, new_entry):
    """
    Appends a new entry to the existing JSON file.
    """
    try:
        with open(file_path, 'r+') as file:
            data = json.load(file)
            data.append(new_entry)
            file.seek(0)
            json.dump(data, file, indent=4)
    except Exception as e:
        print(f"Error saving to JSON: {e}")


def fetch_and_process_memes(json_url, output_file, prompt, nshot, model_name):
    """
    Fetches memes from a JSON file URL, processes each with GPT, and writes results into a JSON file after each generation.
    """
    try:
        # Fetch the JSON data
        response = requests.get(json_url)
        response.raise_for_status()
        memes_data = response.json()

        # Initialize the output file
        with open(output_file, 'w') as f:
            json.dump([], f)

        for i, meme in enumerate(tqdm(memes_data, desc="Processing memes")):            
            title = meme.get("title", "")
            meme_url = meme.get("url", "")
            meme_captions = meme.get("meme_captions", [])

            if i==5:
                break

            # Initialize entry
            meme_entry = {
                "title": title,
                "url": meme_url,
                "meme_captions": meme_captions,
                "gpt_response": ""
            }

            # Skip entries with missing url or meme_captions
            if not meme_url or not meme_captions:
                error_msg = "<<ERROR>> Missing URL or meme_captions."
                print(f"Skipping entry {i + 1}: {error_msg}")
                meme_entry["gpt_response"] = error_msg
                save_to_json(output_file, meme_entry)
                continue

            # Fetch the image from URL
            try:
                img_response = requests.get(meme_url)
                img_response.raise_for_status()
                image = Image.open(BytesIO(img_response.content))
            except Exception as e:
                error_msg = f"<<ERROR>> Failed to fetch image: {e}"
                print(f"Skipping entry {i + 1}: {error_msg}")
                meme_entry["gpt_response"] = error_msg
                save_to_json(output_file, meme_entry)
                continue
            if args.mode != 'prescriptive':
                prompt = prompt.format(title=title)
            else:
                instruction = """
        Using the instruction given, answer the question:
        Question: Following image is a meme with the Title: {title}. Why is this meme funny?
        Make sure that the final answer is followed after 'FinalAnswerWithoutCode:'
                """.format(title=title)
                prompt = prompt + instruction
            # Get response from GPT
            gpt_response = get_response(client, image, prompt=prompt, fewshot_templates=fewshot_templates[:nshot], model_name=model_name)
            if gpt_response.startswith("<<ERROR>>"):
                print(f"Error for entry {i + 1}: {gpt_response}")
            else:
                # print(f"Processed entry {i + 1}: {title}")
                pass

            # Add response to entry
            meme_entry["gpt_response"] = gpt_response

            # Save the entry immediately
            save_to_json(output_file, meme_entry)

        print("Processing complete.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the JSON file: {e}")
    except json.JSONDecodeError:
        print("Error decoding JSON.")

if __name__ == "__main__":
    args = parser.parse_args()
    mode = args.mode
    nshot = args.nshot
    model_name = args.model_name
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    base_url = args.base_url
    key = args.key
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{dataset_name}_{model_name.replace("/","_")}_{mode}_{nshot}_shot.json'
    fewshot_examples = json.load(open("memecap_fewshot_examples.json"))
    fewshot_templates = []
    for f in fewshot_examples:
        sample = [
            { 
            "role" : "user" ,
            "content" : [
                    { "type": "text", "text": "Title:" + f['title'] },
                    { "type": "image_url", "image_url": f['url']},
                    { "type": "text", "text": "The image shows "+"\n".join(f['img_captions']) },
                    { "type": "text", "text": "The causal graph is" + f['CausalGraph'] if mode=='causal' else ""},
                    { "type": "text", "text": "The meme implies " + "\n".join(f['meme_captions']) },
                    { "type": "text", "text": "The metaphors are " + "\n".join([v['metaphor'] + ' implying ' + v['meaning'] for v in f['metaphors']])}
                ]
            }
        ]
        fewshot_templates.extend(sample)
    
    # Initialize OpenAI client
    if base_url is not None and key is not None:
        client = OpenAI(api_key=args.key, base_url=args.base_url)
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)  # Replace with your actual API key
    json_url = "https://raw.githubusercontent.com/eujhwang/meme-cap/refs/heads/main/data/memes-test.json"
    response = requests.get(json_url)
    response.raise_for_status()
    memes_data = response.json()
    prompt = None

    if mode == 'causal':

        prompt = "Following image is a meme with the Title: {title}\nWhy is this meme funny?"
        prompt = prompt + "To answer this, based on the examples given, first create a causal reasoning graph linking different objects, people, entities, text present in the image along with the title given in the form of a piece of code, and then give the final answer. Make sure that the final answer is followed after 'FinalAnswerWithoutCode:'"

    elif mode == 'prescriptive':
        prompt = """
        Step 1: Define Core Elements of Visual Irony
        "First, identify 5-8 key variables (nodes) that are fundamental to detecting irony in images. For each variable, provide an ID and a clear description of what it represents in the context of visual irony."

        Step 2: Establish Primary Causal Relationships
        "Next, determine the direct causal relationships between the variables you identified. For each relationship, specify the source node, target node, and describe exactly how the source influences or affects the target in creating ironic meaning."

        Step 3: Identify Irony Indicators
        "Then, create a special node called 'PerceivedIrony' that represents the determination of whether irony is present. Establish relationships between your previously defined variables and this irony node, explaining specifically how each contributing element leads to the perception of irony."

        Step 4: Define Contextual Influences
        "Identify any contextual or cultural factors that might influence the interpretation of irony in images. Create nodes for these factors and establish their relationships with other elements in your graph."

        Step 5: Format as Structured JSON
        "Finally, organize your complete causal reasoning graph into JSON format. 
        First of all create a list entries with keys nodes and edges. Nodes should contain each node, and its description 
        An edge is comprised of from node, to node, type, strength and explanation.
        Follow this template to construct the causal graph:
        {{
            \"nodes\": [
                {{\"id\": \"node1\", \"name\": \"Variable Name\", \"description\": \"Explanation of this variable\"}}
            ],
            \"edges\": [
                {{\"source\": \"node1\", \"target\": \"node2\", \"type\": \"causal\", \"strength\": \"strong/moderate/weak\", \"explanation\": \"Reasoning for this causal relationship\"}}
            ]
        }}
        """
    else:
        prompt = "Using the exemplars given, answer the question:\n Following image is a meme with the Title: {title}\nWhy is this meme funny?"

    json_url = "https://raw.githubusercontent.com/eujhwang/meme-cap/refs/heads/main/data/memes-test.json"
    fetch_and_process_memes(json_url, output_file, prompt, nshot, model_name)
    print(f"Results saved to {output_file}")

